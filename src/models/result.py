import networkx as nx

from marshmallow import fields
from src.db import db
from src.models.base import BaseModel, BaseSchema
from src.models.node import Node


class Result(BaseModel):
    job_id = db.Column(db.Integer, db.ForeignKey('job.id'), nullable=False)
    job = db.relationship('Job', backref=db.backref('results'))

    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime, nullable=False)
    execution_time = db.Column(db.Float)
    dataset_loading_time = db.Column(db.Float)
    meta_results = db.Column(db.JSON)

    @property
    def ground_truth_statistics(self):
        g1 = nx.DiGraph(id=self.id, name=f'Graph_{self.id}_gt')
        for node in self.job.experiment.dataset.nodes:
            g1.add_node(node.id, label=node.name)
        for node in self.job.experiment.dataset.nodes:
            for edge in node.edge_froms:
                if edge.is_ground_truth:
                    g1.add_edge(edge.from_node.id, edge.to_node.id, id=edge.id, label='', weight=edge.weight)


        ground_truth = nx.DiGraph(id=-1, name=f'Graph_{self.id}_gt')
        for node in self.job.experiment.dataset.nodes:
            ground_truth.add_node(node.id, label=node.name)
        for node in self.job.experiment.dataset.nodes:
            for edge in node.edge_froms:
                if edge.is_ground_truth:
                    ground_truth.add_edge(edge.from_node.id, edge.to_node.id, id=edge.id, label='', weight=1)
        return {'graph_edit_distance': nx.graph_edit_distance(ground_truth,g1)}


class ResultSchema(BaseSchema):
    ground_truth_statistics = fields.Dict()
    class Meta(BaseSchema.Meta):
        exclude = ['edge_informations']
        model = Result
