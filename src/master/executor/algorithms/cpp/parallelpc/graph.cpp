#include "graph.hpp"

#include <vector>
#include <iostream>

Graph::Graph(int num_nodes) {
    _adjacencies = arma::Mat<uint8_t>(num_nodes, num_nodes, arma::fill::ones);
    _adjacencies.diag().zeros();
    _num_nodes = num_nodes;
    _adjacency_lists = std::vector<std::vector<int>>(num_nodes, std::vector<int>());
}

Graph::Graph(Graph &g) {
    g.updateNeighbours();
    _adjacencies = g.getAdjacencies();
    _adjacency_lists = g.getAdjacencyLists();
    _num_nodes = g.getNumberOfNodes();
}

void Graph::deleteEdge(int node_x, int node_y) {
    _adjacencies.at(node_x, node_y) = 0;
    _adjacencies.at(node_y, node_x) = 0;
}

std::vector<int> Graph::getNeighbours(int node_id) const {
    return _adjacency_lists[node_id];
}

std::vector<std::pair<int,int> > Graph::getEdgesToTest() const {
    std::vector<std::pair<int, int> > result;
    for(int i = 0; i < _num_nodes; i++) {
        for(int j = 0; j < i; j++) {
            if(_adjacencies.at(i,j)) {
                result.push_back(std::make_pair(i,j));
            }
        }
    }
    return result;
}

void Graph::print_mat() const {
    _adjacencies.print(std::cout);
}

void Graph::print_list() const {
    for(int i = 0; i < _num_nodes; i++) {
        std::vector<int> adj = getNeighbours(i);
        std::cout << i << " -> ";

        auto begin = adj.begin();
        if (begin != adj.end())
          std::cout << *begin++;
        while (begin != adj.end())
          std::cout << ',' << *begin++;
        std::cout << std::endl;
    }
}

void Graph::updateNeighbours() {
    for (int i = 0; i < _num_nodes; ++i) {
        _adjacency_lists[i].clear();
        for (int j = 0; j < _num_nodes; ++j) {
            if(_adjacencies.at(i,j) && i != j) {
                _adjacency_lists[i].push_back(j);
            }
        }
    }

}

int Graph::getNeighbourCount(int node_id) const {
    return _adjacency_lists[node_id].size();
}

int Graph::getNumberOfNodes() {
    return _num_nodes;
}

arma::Mat<uint8_t> Graph::getAdjacencies() {
    return _adjacencies;
}

std::vector<std::vector<int>> Graph::getAdjacencyLists() {
    return _adjacency_lists;
}

std::vector<int> Graph::getNeighboursWithout(int node_id, int skip) const {
    std::vector<int> result;
    result.reserve(_num_nodes);
    for (int i = 0; i < _num_nodes; ++i)
    {
        if(_adjacencies.at(i,node_id) && i != node_id && i != skip) {
            result.push_back(i);
        }
    }

    return result;
}
