# This script may help you with your development workflow. It only deploys postgres and the scheduler to minikube
# and sets up all port forwarding. After that you can run services/python-images/src/run_dev_server.sh
# and yarn start in services/frontend-image/src
get_host(){
  minikube ssh "cat /etc/hosts | grep host.minikube.internal"| { read -a array ; echo ${array[0]} ; }
}
MINIKUBE_HOST_IP="$(get_host)"
BACKEND_PORT=5000
MPCI_NAMESPACE=mpci-default
garden deploy scheduler --env=minimal-dev-setup --var backendApiHost="$MINIKUBE_HOST_IP:$BACKEND_PORT"
echo "Forwarding Postgres Port"
kubectl port-forward db-postgresql-0 5432 -n $MPCI_NAMESPACE & # this is a StatefulSet so the pod should stay the same
DB_PID=$!
echo "Forwarding Scheduler Port"
kubectl port-forward service/mpci-default-scheduler 4000:80 -n $MPCI_NAMESPACE
SCHEDULER_PID=$!

function clean_up {

    # Perform program exit housekeeping
    kill -2 $DB_PID
    kill -2 $SCHEDULER_PID
    exit
}

trap clean_up SIGHUP SIGINT SIGTERM
wait
