#!/bin/bash

find log -type f

set -eu
function catch {
  echo Catch
}
function finally {
  echo Finally
}
trap catch ERR
trap finally EXIT

read delete_file
rm log/${delete_file}.text results/${delete_file}.png
