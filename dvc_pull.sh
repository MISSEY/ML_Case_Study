#!/bin/bash

extract_git_folder() {

  readarray -d / -t repo_array <<< "$1"
  index="$((${#repo_array[@]} - 1))"
  repo="repo_array[$index]"
  readarray -d . -t repo_folder <<< "${!repo}"

  echo "${repo_folder[0]}"
}

while getopts g::d:v: flag
do
    case "${flag}" in
      g) git_repo=${OPTARG};;
      d) dvc_file=${OPTARG};;
      v) mount_path=${OPTARG};;
      *) echo "paramater not supported"
    esac
done

echo "git_repo: $git_repo"
echo "dvc_file: $dvc_file"
echo "volume: $mount_path"

git_url="https://$GIT_USERNAME:$GIT_TOKEN@$git_repo"

folder=$(extract_git_folder "$git_url")

cd "$mount_path"

pwd

if [ ! -d "$folder" ]; then
  git clone "$git_url"
fi

cd "$folder"
filepath=`pwd`
git config --global --add safe.directory $filepath
git pull origin main

dvc pull -f "$dvc_file"
