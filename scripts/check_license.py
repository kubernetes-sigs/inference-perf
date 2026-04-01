#!/usr/bin/env python3
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

import sys
import datetime

def get_license_template():
    year = datetime.date.today().year
    return f"""# Copyright {year} The Kubernetes Authors.
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
"""

def check_file(filename, template):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.startswith(template):
                print(f"File missing or incorrect license header: {filename}")
                return False
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return False
    return True

def main():
    template = get_license_template()
    failed = False
    for filename in sys.argv[1:]:
        if not check_file(filename, template):
            failed = True
    
    if failed:
        sys.exit(1)
    print("All files passed license check.")

if __name__ == "__main__":
    main()
