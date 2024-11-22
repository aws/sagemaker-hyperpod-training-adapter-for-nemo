# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

#!/bin/bash

check_installed_packages_licenses() {
    echo "Checking Installed Packages Licenses..."
    LICENSE_LIST=$(cat ./ApprovedLicenses.txt | tr '\n' '|'| sed 's/|$//')
    echo "LICENSE_LIST: $LICENSE_LIST"
    pip-licenses --summary > LicenseSummary.txt
    awk '{$1=""; print $0}' ./LicenseSummary.txt | tail -n +2 | sed 's/;/\n/g' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'| sort -u > ./newLicenseSummary.txt
    cat ./newLicenseSummary.txt
    while IFS= read -r line || [[ -n "$line" ]]; do
        if ! echo "$LICENSE_LIST" | grep -q "$line"; then
            echo "License '$line' is not in the allowed list."
            exit 1
        fi
    done < ./newLicenseSummary.txt
    echo "Installed Packages License Check complete"
}
check_source_code_licenses() {
    echo "Checking Source Code Licenses..."
    if ! grep -q "prohibited-license: Did not find content matching specified patterns" ./scanOutput.txt; then
        echo "Prohibited License Used in Source Code Scan: "
        sed -n '/⚠  prohibited-license:/,/⚠  third-party-license-file:/p' ./scanOutput.txt | sed '1d;$d'| cat
        exit 1
    fi
    echo "Source Code License Check complete"
}

case $1 in
    source_code)
        check_source_code_licenses
        ;;
    installed_packages)
        check_installed_packages_licenses
        ;;
    *)
        echo "Invalid argument. Please provide either 'source_code' or 'installed_packages'."
        ;;
esac
