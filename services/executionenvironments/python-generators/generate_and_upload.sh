set -e
OTHER_PARAMS=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --uploadEndpoint)
    UPLOAD_ENDPOINT="$2"
    shift # past argument
    shift # past value
    ;;
    --apiHost)
    API_HOST="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    OTHER_PARAMS+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
echo "host: ${API_HOST} endpoint: ${UPLOAD_ENDPOINT}"
python src/__main__.py ${OTHER_PARAMS[@]}
python upload_results.py --apiHost $API_HOST --uploadEndpoint $UPLOAD_ENDPOINT
