#!/bin/bash

# Create the .git/hooks directory if it doesn't exist
mkdir -p .git/hooks

# Create the pre-commit file
cat << 'EOF' > .git/hooks/pre-commit
###################################### SCRIPT ##########################################

#!/bin/bash

echo "Running pre-commit hook..."

# Function to create an archive for a given directory
create_archive() {
    local dir=$1
    local include_file="$dir/.include"
    local archive_name="$dir.zip"

    if [ -f "$include_file" ]; then
        echo "Creating archive for directory: $dir"

        # Create a temporary directory to store the files to be archived
        temp_dir=$(mktemp -d)

        # Copy the files listed in the .include file to the temporary directory
        while IFS= read -r file; do
            if [ -e "$dir/$file" ]; then
                mkdir -p "$temp_dir/$(dirname "$file")"
                cp -r "$dir/$file" "$temp_dir/$file"
            else
                echo "File or directory not found: $dir/$file"
            fi
        done < "$include_file"

        # Create the archive in the original directory
        (cd "$temp_dir" && zip -r "$archive_name" .)

        # Move the archive to the original directory
        mv "$temp_dir/$archive_name" "$dir/"

        # Clean up the temporary directory
        rm -rf "$temp_dir"
    else
        echo "No .include file found in directory: $dir"
    fi
}

# Find all directories containing an .include file and create archives
find . -type f -name ".include" -exec dirname {} \; | while read -r dir; do
    create_archive "$dir"
done

# Allow the commit to proceed
exit 0

###################################### SCRIPT ##########################################
EOF


# Make the pre-commit file executable
chmod +x .git/hooks/pre-commit

echo "pre-commit hook has been set up successfully."