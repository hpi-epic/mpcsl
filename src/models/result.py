import networkx as nx

from marshmallow import fields
from src.db import db
from src.models.base import BaseModel, BaseSchema


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
        from src.master.helpers.database import load_networkx_graph
        g1 = load_networkx_graph(self)

        ground_truth = nx.DiGraph(id=-1, name=f'Graph_{self.id}_gt')
        for node in self.job.experiment.dataset.nodes:
            ground_truth.add_node(node.id, label=node.name)
        for node in self.job.experiment.dataset.nodes:
            for edge in node.edge_froms:
                if edge.is_ground_truth:
                    ground_truth.add_edge(edge.from_node.id, edge.to_node.id, id=edge.id, label='', weight=1)

        ground_truth_statistics = {}
        ground_truth_statistics['graph_edit_distance'] = nx.graph_edit_distance(ground_truth, g1)
        ground_truth_statistics['jaccard_coefficients'] = Result.get_jaccard_coefficients(g1, ground_truth)
        return ground_truth_statistics

    @staticmethod
    def get_jaccard_coefficients(G, H):
        jc = []
        for v in G:
            n = set(G[v])
            m = set(H[v])
            length_intersection = len(n & m)
            length_union = len(n) + len(m) - length_intersection
            if length_union != 0:
                jc.append((v, float(length_intersection) / length_union))
            else:
                jc.append((v, 1.0))
        return jc


class ResultSchema(BaseSchema):
    ground_truth_statistics = fields.Dict()

    class Meta(BaseSchema.Meta):
        exclude = ['edge_informations']
        model = Result
