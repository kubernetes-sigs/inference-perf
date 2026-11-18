# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Literal, Tuple, Type

import pytest
from pydantic import ValidationError

from inference_perf.workload import (
    AudioPart,
    AudioSpec,
    Delivery,
    ImagePart,
    ImageSpec,
    Message,
    ReasoningPart,
    Record,
    Schedule,
    Session,
    TextPart,
    TextSpec,
    Tool,
    ToolCallPart,
    ToolResultPart,
    VideoPart,
    VideoSpec,
    continues,
)
from inference_perf.workload import record as schema
from inference_perf.workload.record import StrictModel


# One request "q" asking for 3 output tokens, called `record_id`.
def request(record_id: str) -> Record:
    return Record(id=record_id, messages=[Message(role="user", parts=[TextPart(text="q")])], output_tokens=3)


# The session "s" of `records`. `waits` names, per record id, the records it
# waited for: {"b": ["a"]} is b sent after a.
def session(records: List[Record], waits: Dict[str, List[str]] | None = None) -> Session:
    schedules = {record_id: Schedule(after=after) for record_id, after in (waits or {}).items()}
    return Session(id="s", records=records, delivery=Delivery(schedules=schedules))


# Every class of the schema, found by walking the module.
def schema_classes() -> List[Type[StrictModel]]:
    found = [value for value in vars(schema).values() if isinstance(value, type) and issubclass(value, StrictModel)]
    return [cls for cls in found if cls is not StrictModel]


# Every class has a docstring and every field a description, so the schema
# reads as one line per thing it can record. `type` is exempt: it only names
# the class a part is. The walk finds all seventeen classes, Record among them.
def test_every_class_and_field_says_what_it_records() -> None:
    classes = schema_classes()
    assert Record in classes and Schedule in classes and len(classes) == 17
    for cls in classes:
        assert (cls.__doc__ or "").strip(), f"{cls.__name__} has no docstring"
        for name, field in cls.model_fields.items():
            if name == "type":
                continue
            assert (field.description or "").strip(), f"{cls.__name__}.{name} has no description"


# A text spec of 1000 tokens with 512-token blocks has room for two blocks
# (the second one partial). Three block ids are rejected; two are accepted
# and cover min(2 * 512, 1000) = 1000 prefix tokens, one covers 512.
def test_text_spec_blocks_must_fit() -> None:
    with pytest.raises(ValidationError, match="cannot fit"):
        TextSpec(num_tokens=1000, block_ids=[1, 2, 3], block_size=512)
    assert TextSpec(num_tokens=1000, block_ids=[1, 2], block_size=512).prefix_tokens == 1000
    assert TextSpec(num_tokens=1000, block_ids=[1], block_size=512).prefix_tokens == 512


# A spec does not guess its block size: leaving it out is rejected.
def test_text_spec_needs_a_block_size() -> None:
    with pytest.raises(ValidationError, match="block_size"):
        TextSpec.model_validate({"num_tokens": 10})


# Text is stated as recorded or as a spec to build, never both and never
# neither. The same rule holds for an image, audio and a video with `ref`.
def test_a_part_is_recorded_or_built_and_not_both() -> None:
    spec = TextSpec(num_tokens=4, block_size=2)
    assert TextPart(text="hi").spec is None
    assert TextPart(spec=spec).text is None
    with pytest.raises(ValidationError, match="exactly one of `text` and `spec`"):
        TextPart()
    with pytest.raises(ValidationError, match="exactly one of `text` and `spec`"):
        TextPart(text="hi", spec=spec)
    assert ImagePart(ref="file:///cat.png").spec is None
    assert ImagePart(spec=ImageSpec(width=64, height=32)).ref is None
    assert AudioPart(spec=AudioSpec(seconds=1.5)).ref is None
    assert VideoPart(spec=VideoSpec(width=64, height=32, frames=8)).ref is None
    with pytest.raises(ValidationError, match="exactly one of `ref` and `spec`"):
        ImagePart()
    with pytest.raises(ValidationError, match="exactly one of `ref` and `spec`"):
        ImagePart(ref="file:///cat.png", spec=ImageSpec(width=64, height=32))
    with pytest.raises(ValidationError, match="exactly one of `ref` and `spec`"):
        AudioPart()
    with pytest.raises(ValidationError, match="exactly one of `ref` and `spec`"):
        AudioPart(ref="file:///a.wav", spec=AudioSpec(seconds=1.5))
    with pytest.raises(ValidationError, match="exactly one of `ref` and `spec`"):
        VideoPart()
    with pytest.raises(ValidationError, match="exactly one of `ref` and `spec`"):
        VideoPart(ref="file:///v.mp4", spec=VideoSpec(width=64, height=32, frames=8))


# A record is one request: its messages are what it sends and its reply is
# what the recording got back. A record with no messages is rejected. A
# message is from the system, the user or the assistant: a "tool" role is
# rejected, since a tool result is a part. A record carries no timing: giving
# it a schedule is an unknown key.
def test_a_record_is_what_one_request_carries() -> None:
    hi = Message(role="user", parts=[TextPart(text="hi")])
    record = Record(id="r", messages=[hi], output_tokens=1, reply=[TextPart(text="hello")])
    assert record.messages == [hi] and record.reply == [TextPart(text="hello")]
    assert Record(id="bare", messages=[hi]).reply == []
    with pytest.raises(ValidationError):
        Record(id="empty", messages=[])
    with pytest.raises(ValidationError):
        Message.model_validate({"role": "tool", "parts": []})
    with pytest.raises(ValidationError, match="schedule"):
        Record.model_validate({**record.model_dump(mode="json"), "schedule": {"start_ms": 0}})


# The assistant's reply calls read_file as call "c1", and the next request
# sends that reply back followed by the file as a tool result answering "c1".
# The result is accepted in the second record because the call sits before
# it. A result answering "c9", which nothing called, is rejected, and so is a
# result placed before its call. Calls and results without ids are accepted.
def test_a_tool_result_answers_an_earlier_call() -> None:
    ask = Message(role="user", parts=[TextPart(text="fix the bug")])
    call = ToolCallPart(id="c1", name="read_file", arguments=TextPart(text='{"path": "a.py"}'))
    result = ToolResultPart(call_id="c1", content=[TextPart(text="print(1)")])
    tools = [Tool(name="read_file", description="Read a file", parameters={"type": "object"})]
    first = Record(id="r1", messages=[ask], tools=tools, reply=[call])
    second = Record(
        id="r2",
        messages=[ask, Message(role="assistant", parts=[call]), Message(role="user", parts=[result])],
        tools=tools,
    )
    assert first.reply == [call] and second.messages[2].parts == [result]
    with pytest.raises(ValidationError, match="answers 'c9', which no earlier call made"):
        Record(id="r", messages=[ask, Message(role="user", parts=[ToolResultPart(call_id="c9")])])
    with pytest.raises(ValidationError, match="answers 'c1', which no earlier call made"):
        Record(id="r", messages=[Message(role="user", parts=[result]), Message(role="assistant", parts=[call])])
    unnamed = Record(
        id="r",
        messages=[ask, Message(role="assistant", parts=[ToolCallPart()]), Message(role="user", parts=[ToolResultPart()])],
    )
    assert unnamed.messages[2].parts == [ToolResultPart()]
    # a server-run tool: the call and its result both sit in the reply
    served = Record(id="r", messages=[ask], reply=[call, result])
    assert served.reply == [call, result]


# The main agent asks "find the bug" and its reply calls a search agent as
# call "c7". A record with the given id, sent to carry out `call_id`.
def spawning(record_id: str, call_id: str | None) -> List[Record]:
    ask = Message(role="user", parts=[TextPart(text="find the bug")])
    main = Record(id="main", messages=[ask], reply=[ToolCallPart(id="c7", name="search_agent")])
    look = Message(role="user", parts=[TextPart(text="look in a.py")])
    return [main, Record(id=record_id, messages=[look], agent="search", call_id=call_id)]


# The subagent's request names "c7", the call the main agent's reply made, so
# the session is accepted and the request says which call it carries out.
def test_a_subagent_request_names_the_call_that_started_it() -> None:
    made = session(spawning("sub", "c7"))
    assert [record.call_id for record in made.records] == [None, "c7"]


# The subagent's request names "c9", which no reply in the session made.
def test_a_request_cannot_carry_out_a_call_nothing_made() -> None:
    with pytest.raises(ValidationError, match="record 'sub' carries out 'c9', which no other record's reply made"):
        session(spawning("sub", "c9"))


# A record whose own reply makes call "c7" says it carries out "c7". A request
# cannot be started by a call it makes itself, so the session is rejected.
def test_a_request_cannot_carry_out_its_own_call() -> None:
    ask = Message(role="user", parts=[TextPart(text="find the bug")])
    own = Record(id="main", messages=[ask], reply=[ToolCallPart(id="c7", name="search_agent")], call_id="c7")
    with pytest.raises(ValidationError, match="record 'main' carries out 'c7', which no other record's reply made"):
        session([own])


# A reply of 700 tokens of thinking and then 200 of answer is two parts, so
# the two lengths stay apart. Thinking holds content the way a tool result
# does: an image is accepted beside text. A tool call inside it is rejected.
def test_reasoning_is_kept_apart_from_the_answer() -> None:
    thought = TextPart(spec=TextSpec(num_tokens=700, block_size=64))
    answer = TextPart(spec=TextSpec(num_tokens=200, block_size=64))
    ask = Message(role="user", parts=[TextPart(text="draw a cat")])
    record = Record(id="r", messages=[ask], reply=[ReasoningPart(content=[thought]), answer])
    assert [part.type for part in record.reply] == ["reasoning", "text"]
    sketch = ReasoningPart(content=[thought, ImagePart(ref="file:///sketch.png")])
    assert [inner.type for inner in sketch.content] == ["text", "image"]
    with pytest.raises(ValidationError):
        ReasoningPart.model_validate({"content": [{"type": "tool_call", "name": "look"}]})


# Dumping a session of one record that holds every kind of part, delivered
# at 1500 ms, and validating the dump, gives back the same session with each
# part's class restored from `type`. The timing is in the dump under the
# session's delivery and nowhere in the record.
def test_session_round_trips_through_dump() -> None:
    call = ToolCallPart(id="c1", name="look")
    record = Record(
        id="r",
        model="m",
        agent="main",
        output_tokens=20,
        metadata={"origin": "test"},
        settings={"temperature": 0.2, "tool_choice": "required"},
        tools=[Tool(name="look")],
        messages=[
            Message(
                role="user",
                parts=[
                    TextPart(text="describe"),
                    TextPart(spec=TextSpec(num_tokens=600, block_ids=[7, 8], block_size=512, scope="session")),
                    ImagePart(ref="file:///cat.png"),
                    AudioPart(spec=AudioSpec(seconds=2.0)),
                    VideoPart(spec=VideoSpec(width=64, height=32, frames=8)),
                ],
            ),
            Message(role="assistant", parts=[call]),
            Message(role="user", parts=[ToolResultPart(call_id="c1", content=[ImagePart(ref="file:///dog.png")])]),
        ],
        reply=[ReasoningPart(content=[TextPart(spec=TextSpec(num_tokens=700, block_size=64))]), TextPart(text="a cat")],
    )
    delivery = Delivery(schedules={"r": Schedule(start_ms=1500, duration_ms=300, wait_ms=20)})
    whole = Session(id="s", records=[record], delivery=delivery)
    dumped = whole.model_dump(mode="json")
    restored = Session.model_validate(dumped)
    assert restored == whole
    kinds = [p.type for p in restored.records[0].messages[0].parts]
    assert kinds == ["text", "text", "image", "audio", "video"]
    assert type(restored.records[0].messages[1].parts[0]) is ToolCallPart
    assert type(restored.records[0].messages[2].parts[0]) is ToolResultPart
    assert type(restored.records[0].reply[0]) is ReasoningPart
    assert restored.records[0].settings == {"temperature": 0.2, "tool_choice": "required"}
    assert dumped["delivery"]["schedules"]["r"] == {"start_ms": 1500, "duration_ms": 300, "after": [], "wait_ms": 20}
    assert sorted(dumped["records"][0]) == [
        "agent",
        "call_id",
        "id",
        "messages",
        "metadata",
        "model",
        "output_tokens",
        "reply",
        "settings",
        "tools",
    ]


# A record the delivery says nothing about has no recorded time and waited
# for nothing. A session with its delivery dropped keeps the same records and
# no record waits for another, which is what a paced run sends.
def test_a_session_without_a_delivery_is_just_its_records() -> None:
    chain = session([request("a"), request("b")], {"b": ["a"]})
    assert chain.delivery.of("a") == Schedule()
    assert chain.delivery.of("b").after == ["a"]
    paced = chain.model_copy(update={"delivery": Delivery()})
    assert paced.records == chain.records
    assert [paced.delivery.of(record.id).after for record in paced.records] == [[], []]


# The schema is strict: a part with a key it does not define is rejected
# rather than silently dropped.
def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        TextPart.model_validate({"type": "text", "text": "x", "tokens": 3})


# Two records with the same id, a record waiting for an id the session does
# not have, and a delivery that schedules an id the session does not have are
# rejected, and so is a session with no records.
def test_session_ids_and_waits() -> None:
    with pytest.raises(ValidationError, match="unique"):
        session([request("a"), request("a")])
    with pytest.raises(ValidationError, match="waits for 'zzz', which is not in the session"):
        session([request("a")], {"a": ["zzz"]})
    with pytest.raises(ValidationError, match="schedules 'zzz', which is not in the session"):
        session([request("a")], {"zzz": []})
    with pytest.raises(ValidationError):
        session([])


# A record that waits for itself, two records that wait for each other, and
# a cycle hanging off a valid root (r <- a <-> b) are all rejected. A diamond
# (a before b and c, both before d) is not a cycle and is accepted.
def test_session_rejects_wait_cycles() -> None:
    with pytest.raises(ValidationError, match="cycle.*'a'"):
        session([request("a")], {"a": ["a"]})
    with pytest.raises(ValidationError, match=r"cycle.*\['a', 'b'\]"):
        session([request("a"), request("b")], {"a": ["b"], "b": ["a"]})
    with pytest.raises(ValidationError, match=r"cycle.*\['a', 'b'\]"):
        session([request("r"), request("a"), request("b")], {"a": ["r", "b"], "b": ["a"]})
    diamond = session([request("a"), request("b"), request("c"), request("d")], {"b": ["a"], "c": ["a"], "d": ["b", "c"]})
    assert diamond.delivery.of("d").after == ["b", "c"]


Role = Literal["system", "user", "assistant"]


# A record that sends the given (role, text) messages and recorded `reply`
# as its answer, asking for 3 output tokens.
def talk(record_id: str, messages: List[Tuple[Role, str]], reply: str) -> Record:
    built = [Message(role=role, parts=[TextPart(text=text)]) for role, text in messages]
    return Record(id=record_id, messages=built, output_tokens=3, reply=[TextPart(text=reply)])


# a asks "q1" and is answered "r1". b waited for a and sends a's messages,
# a's reply, then "q2". c waited for b and adds "q3". So b continues a, c
# continues b (the longer conversation, not a), and a continues nothing.
def test_a_record_continues_the_longest_conversation_it_waited_for() -> None:
    a = talk("a", [("user", "q1")], "r1")
    b = talk("b", [("user", "q1"), ("assistant", "r1"), ("user", "q2")], "r2")
    c = talk("c", [("user", "q1"), ("assistant", "r1"), ("user", "q2"), ("assistant", "r2"), ("user", "q3")], "")
    chain = session([a, b, c], {"b": ["a"], "c": ["b"]})
    assert continues(chain, a) is None
    assert continues(chain, b) is a
    assert continues(chain, c) is b
    # naming both waits, in either order, still picks the longer one
    for after in (["a", "b"], ["b", "a"]):
        assert continues(session([a, b, c], {"b": ["a"], "c": after}), c) is b


# b carries a's whole conversation but never waited for a, so a's live reply
# may not exist when b is sent: b continues nothing. With the wait it does.
# The records are the same in all three sessions, only the delivery differs.
def test_a_record_only_continues_one_it_waited_for() -> None:
    a = talk("a", [("user", "q1")], "r1")
    b = talk("b", [("user", "q1"), ("assistant", "r1"), ("user", "q2")], "r2")
    assert continues(session([a, b]), b) is None
    assert continues(session([a, b], {"b": ["a"]}), b) is a
    # waiting through an unrelated record x counts: x waited for a, b for x
    x = talk("x", [("user", "elsewhere")], "ok")
    assert continues(session([a, x, b], {"x": ["a"], "b": ["x"]}), b) is a


# b waited for a, but it carries a different reply ("other" where a was
# answered "r1"), then a different role in the same place, then only a's
# question with a's answer as its own reply. None of the three continues a.
def test_a_record_with_a_different_opening_continues_nothing() -> None:
    a = talk("a", [("user", "q1")], "r1")
    other_reply = talk("b", [("user", "q1"), ("assistant", "other"), ("user", "q2")], "r2")
    other_role = talk("b", [("user", "q1"), ("user", "r1"), ("user", "q2")], "r2")
    too_short = talk("b", [("user", "q1")], "r1")
    for b in (other_reply, other_role, too_short):
        assert continues(session([a, b], {"b": ["a"]}), b) is None


# b sends exactly a's messages and a's reply, with nothing after them: b
# still continues a, and has no new messages of its own to send.
def test_a_record_may_continue_another_and_add_nothing() -> None:
    a = talk("a", [("user", "q1")], "r1")
    b = talk("b", [("user", "q1"), ("assistant", "r1")], "r2")
    assert continues(session([a, b], {"b": ["a"]}), b) is a
    assert b.messages[len(a.messages) + 1 :] == []
