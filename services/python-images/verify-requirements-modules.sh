# checks all implicit dependencies are listed in requirements.txt
# if not, some implicit dependencies may be updated without our knowledge, which led to error in the past
pip freeze > tmp_requirements.txt
DIFF=$(diff tmp.txt requirements.txt)
if [[ $DIFF ]]; then
    echo "Some dependencies are not in requirements.txt"
    echo "$DIFF"
    exit 1
else
    echo "All dependencies are listed in requirements.txt"
fi

