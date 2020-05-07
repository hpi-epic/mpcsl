### Kubernetes Installation

[Installation Guide](https://thenewstack.io/how-to-deploy-a-kubernetes-cluster-with-ubuntu-server-18-04/)

[Important](https://coreos.com/flannel/docs/latest/kubernetes.html): Choose the same CIDR for the pods in `sudo kubeadm init --pod-network-cidr=<CIDR>` and the flannel network.
Flannels default CIDR is `10.244.0.0/16`.

### Cluster Deployments

For a valid running cluster a storage provisioner is needed.
Run `helm install nfs-client stable/nfs-client-provisioner -f nfs-client-values.yml` to deploy the provisioner into the cluster.

For the ingress routing some ingress controller is needed.
Run `helm install nginx-ingress stable/nginx-ingress -f nfs-client-values.yml` to deploy 

### Further configuration

## Remove master taints
So that pods can be scheduled on the master node `kubectl taint nodes vm-mpws2019 node-role.kubernetes.io/master:NoSchedule-` has to be executed. (Here is vm-mpws2019 our master)

## Add nvidia kubernetes plugin

To add nvidia cuda compatibility to the cluster follow the [installation guide](https://github.com/NVIDIA/k8s-device-plugin).
After enabling the node to use nvidia docker images `kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/1.0.0-beta4/nvidia-device-plugin.yml` has to be executed.

### Deploy the application

Deploy the applciation with:
`garden deploy --env=<ENV> --yes`
After that you can add the algorithms with:
`garden run task db-setup-algorithms --env=<ENV>`
To seed example data in the db run:
`garden run task seed-db --env=<ENV>`

### Add new external ip to ingress
If another external IP has to be accessible, update the following file `deployment-resources/nginx-ingress-values.yml` and execute `helm upgrade nginx-ingress -f deployment-resources/nginx-ingress-values.yml -n default stable/nginx-ingress`