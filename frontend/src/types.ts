export type d3Node = {
  id: string;
  group: number;
  gfx?: PIXI.Graphics;
  x?: number;
  y?: number;
};

export type d3Link = {
  source: string;
  target: string;
  value: number;
};

export type Graph = {
  nodes: d3Node[];
  links: d3Link[];
};
