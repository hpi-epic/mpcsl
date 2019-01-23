#pragma once

#include <armadillo>
#include <string>
#include <vector>
#include <memory>

class Graph {
public:
    Graph(int num_nodes);
    Graph(Graph &g);

    void deleteEdge(int node_x, int node_y);
    std::vector<int> getNeighbours(int node_id) const;
    std::vector<int> getNeighboursWithout(int node_id, int skip) const;
    int getNeighbourCount(int node_id) const;
    arma::Mat<uint8_t> getAdjacencies();
    std::vector<std::vector<int>> getAdjacencyLists();
    std::vector<int> getNeighbourVector();
    void updateNeighbours();
    int getNumberOfNodes();

    std::vector<std::pair<int,int>> getEdgesToTest() const;
    void print_list() const;
    void print_mat() const;



protected:
    arma::Mat<uint8_t> _adjacencies;
    std::vector<std::vector<int>> _adjacency_lists;
    int _num_nodes;

};
