import React, { Component, Fragment } from 'react';
import './App.css';
import * as dagre from 'dagre';
import GraphVis from 'react-digraph-editor';

class App extends Component<any, any>{
  graph: dagre.graphlib.Graph;

  constructor(props: any) {
    super(props);
    this.state = {
      isVisible: true
    }

    this.graph = new dagre.graphlib.Graph();
    this.generateGraph();
  }

  generateGraph = () => {
    this.graph.setGraph({});

    // Default to assigning a new object as a label for each new edge.
    this.graph.setDefaultEdgeLabel(function() {
      return {};
    });

    // Add nodes to the graph. The first argument is the node id. The second is
    // metadata about the node. In this case we're going to add labels to each of
    // our nodes.
    this.graph.setNode('kspacey', { label: 'Kevin Spacey', width: 144, height: 100 });
    this.graph.setNode('swilliams', { label: 'Saul Williams', width: 160, height: 100 });
    this.graph.setNode('bpitt', { label: 'Brad Pitt', width: 108, height: 100 });
    this.graph.setNode('hford', { label: 'Harrison Ford', width: 168, height: 100 });
    this.graph.setNode('lwilson', { label: 'Luke Wilson', width: 144, height: 100 });
    this.graph.setNode('kbacon', { label: 'Kevin Bacon', width: 121, height: 100 });

    // Add edges to the graph.
    this.graph.setEdge('kspacey', 'swilliams');
    this.graph.setEdge('swilliams', 'kbacon');
    this.graph.setEdge('bpitt', 'kbacon');
    this.graph.setEdge('hford', 'lwilson');
    this.graph.setEdge('lwilson', 'kbacon');

    // layout the graph
    dagre.layout(this.graph);
  }

  hideGraph = () => {
    this.setState({ isVisible: !this.state.isVisible });
    console.log(this.graph);
  }

  render() {
    let config = {
      width: screen.width,
      height: 500,
      antialias: true,
      transparent: false,
      resolution: 2,
      autoResize: true,
      graph: this.graph
    };

    console.log('test');

    return (
      <div className="App">
        {this.state.isVisible ? <GraphVis {...config}/> : null}
        <button onClick={this.hideGraph}>Hide Graph</button>
      </div>
    );
  }
}

export default App;
