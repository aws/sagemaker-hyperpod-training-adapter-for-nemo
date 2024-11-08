#!/bin/bash
check_licenses() {
    LICENSE_LIST=$(cat ./ApprovedLicenses.txt | tr '\n' '|'| sed 's/|$//')
    echo "LICENSE_LIST: $LICENSE_LIST"
    pip-licenses --summary > LicenseSummary.txt
    cat ./LicenseSummary.txt
    awk '{$1=""; print $0}' ./LicenseSummary.txt | tail -n +2 | sed 's/;/\n/g' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'| sort -u > ./newLicenseSummary.txt
    cat ./newLicenseSummary.txt
    while IFS= read -r line || [[ -n "$line" ]]; do
        echo "Text read from file: $line"
        if ! echo "$LICENSE_LIST" | grep -q "$line"; then
            echo "License '$line' is not in the allowed list."
            exit 1
        fi
    done < ./newLicenseSummary.txt

    echo "License Check complete"
}

check_licenses
