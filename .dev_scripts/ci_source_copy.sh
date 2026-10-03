#!/bin/bash
set -e

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 SOURCE_DIR COMMAND [ARG...]" >&2
  exit 2
fi

source_dir=$1
shift
test_dir="$(mktemp -d /tmp/swift-ci.XXXXXX)"
trap 'rm -rf -- "${test_dir:?}"' EXIT

# Build products and Git changes stay in the container, owned by its test user.
cp -R "$source_dir/." "$test_dir/"
cd "$test_dir"
"$@"
