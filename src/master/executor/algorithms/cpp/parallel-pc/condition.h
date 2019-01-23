#pragma once

 /**
 * Auxiliary methods for constraint-based algorithms
 *
 * @author Alain Hauser <alain.hauser@biology.unibe.ch>
 * $Id: $
 */

#include <armadillo>
#include <vector>
#include <boost/graph/adjacency_list.hpp>

// Define BGL class for undirected graph
typedef boost::adjacency_list<boost::setS, boost::vecS, boost::undirectedS> InternalUndirectedGraph;
typedef boost::graph_traits<InternalUndirectedGraph>::out_edge_iterator UndirOutEdgeIter;
typedef boost::graph_traits<InternalUndirectedGraph>::edge_iterator UndirEdgeIter;

// Define type for specification of separation sets
typedef std::vector<std::vector<arma::ivec > > SepSets;

/**
 * Virtual base class for conditional independence tests
 */
class IndepTest
{
public:
    /**
     * Virtual destructor
     *
     * Needed because of different storage of data.
     */
    virtual ~IndepTest() {}

    IndepTest() = default;

    /**
     * Virtual test function: must be implemented in all derived classes
     *
     * @param	u	First variable index
     * @param	v	Second variable index
     * @param	S	Conditioning set
     * @return	p-value for independence test of X_u and X_v given X_S
     */
    virtual double test(int u, int v, std::vector<int> &S) const = 0;
};

/**
 * Conditional independence test for Gaussian data
 */
class IndepTestGauss : public IndepTest
{
protected:
    /**
     * Sufficient statistic for easier access in independence test
     */
    int _sampleSize;
    arma::mat _correlation;

public:

    IndepTestGauss() = default;
    
    IndepTestGauss(int sampleSize, arma::Mat<double>& cor) :
            _sampleSize(sampleSize),
            _correlation(cor) {}

    virtual double test(int u, int v, std::vector<int> &S) const;
};
