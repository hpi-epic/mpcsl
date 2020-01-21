import networkx as nx

from marshmallow import fields
from src.db import db
from src.models.base import BaseModel, BaseSchema


class Result(BaseModel):
    job_id = db.Column(db.Integer, db.ForeignKey('job.id'), nullable=False)
    job = db.relationship('Job', backref=db.backref('results', cascade="all, delete-orphan"))

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

        if ground_truth.edges():
            jaccard_coefficients = Result.get_jaccard_coefficients(g1, ground_truth)
            error_types = Result.get_error_types(g1, ground_truth)
            ground_truth_statistics = {
                'graph_edit_distance': nx.graph_edit_distance(ground_truth, g1),
                'mean_jaccard_coefficient':
                    sum(jaccard_coefficients) / len(jaccard_coefficients) if jaccard_coefficients else 0,
                'error_types': error_types
            }
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
                jc.append(float(length_intersection) / length_union)
            else:
                jc.append(1.0)
        return jc

    @staticmethod
    def get_error_types(g1, ground_truth):
        g1_edges_set = set(g1.edges())
        g1_non_edges_set = set(nx.non_edges(g1))
        ground_truth_edges_set = set(ground_truth.edges())
        ground_truth_non_edges_set = set(nx.non_edges(ground_truth))

        false_negatives = list(ground_truth_edges_set & g1_non_edges_set)
        true_positives = list(ground_truth_edges_set & g1_edges_set)
        false_positives = list(ground_truth_non_edges_set & g1_edges_set)
        true_negatives = list(ground_truth_non_edges_set & g1_non_edges_set)

        try:
            false_positives_rate = len(false_positives)/(len(false_positives) + len(true_negatives))
        except ZeroDivisionError:
            false_positives_rate = 0

        try:
            false_negatives_rate = len(false_negatives)/(len(false_negatives) + len(true_positives))
        except ZeroDivisionError:
            false_negatives_rate = 0

        error_types = {
            'false_positives': {'rate': false_positives_rate, 'edges': false_positives},
            'true_negatives': {'rate': 1-false_positives_rate, 'edges': true_negatives},
            'false_negatives': {'rate': false_negatives_rate, 'edges': false_negatives},
            'true_positives': {'rate': 1-false_negatives_rate, 'edges': true_positives}
        }

        return error_types


class ResultSchema(BaseSchema):
    ground_truth_statistics = fields.Dict()

    class Meta(BaseSchema.Meta):
        exclude = ['edge_informations']
        model = Result
