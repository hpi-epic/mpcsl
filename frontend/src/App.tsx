import React, { Component, Fragment } from 'react';
import * as PIXI from 'pixi.js';
import * as d3 from 'd3';
import * as d3force from 'd3-force';
import './GraphData.ts';
import './App.css';
import testData from './GraphData';
import { Graph, d3Node, d3Link } from './types';

interface IMainState {}
interface IMainProps {}
class App extends Component<IMainProps, IMainState> {
  graphCanvas: HTMLDivElement | null;

  constructor(props: any) {
    super(props);
    this.graphCanvas = null;
  }

  componentDidMount() {
    let width: number = 960;
    let height: number = 600;

    let stage = new PIXI.Container();
    let render = new PIXI.Application({
      width,
      height,
      antialias: true,
      transparent: false,
      resolution: 1,
      backgroundColor: 0xfffffff
    });

    let color = (function() {
      let scale: any = d3.scaleOrdinal(d3.schemeCategory10);
      return (num: number) => parseInt(scale(num).slice(1), 16);
    })();

    let simulation = d3
      .forceSimulation()
      .force('link', d3.forceLink().id((d: any) => d.id))
      .force('charge', d3.forceManyBody())
      .force('center', d3.forceCenter(width / 2, height / 2));

    let links = new PIXI.Graphics();
    stage.addChild(links);

    testData.nodes.forEach((node: d3Node) => {
      node.gfx = new PIXI.Graphics();
      node.gfx.lineStyle(1.5, 0xffffff);
      node.gfx.beginFill(color(node.group));
      node.gfx.drawCircle(0, 0, 5);
      stage.addChild(node.gfx);
    });


    simulation.nodes(testData.nodes).on('tick', ticked);

    function ticked() {
      testData.nodes.forEach(node => {
        let { x, y, gfx } = node;
        gfx!.position = new PIXI.Point(x, y);
      });

      links.clear();
      links.alpha = 0.6;

      testData.links.forEach((link: d3Link) => {
        let { source, target } = link;
        links.lineStyle(Math.sqrt(link.value), 0x999999);
        //  @ts-ignore
        links.moveTo(source.x, source.y).lineTo(target.x, target.y);
      });
      links.endFill();
    };

    simulation
      .force<d3force.ForceLink<any, d3Link>>('link')!
      .links(testData.links);

    this.graphCanvas!.appendChild(render.view);
  }

  render() {
    let component = this;
    return (
      <div
        ref={thisDiv => {
          component.graphCanvas = thisDiv;
        }}
      />
    );
  }
}

export default App;
