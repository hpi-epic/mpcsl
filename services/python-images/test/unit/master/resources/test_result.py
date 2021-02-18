from src.db import db
from src.master.resources.results import ResultListResource, ResultResource, ResultCompareResource
from src.models import Node, Result, Edge
from test.factories import ResultFactory, NodeFactory, EdgeFactory, SepsetFactory
from .base import BaseResourceTest


class ResultTest(BaseResourceTest):
    def test_returns_all_results(self):
        # Given
        result = ResultFactory()
        result2 = ResultFactory()

        # When
        results = self.get(self.url_for(ResultListResource))

        # Then
        assert len(results) == 2
        assert results[0]['id'] == result.id
        assert results[1]['id'] == result2.id

    def test_returns_my_job(self):
        # Given
        result = ResultFactory()
        nodes = [NodeFactory(dataset=result.job.experiment.dataset) for _ in range(3)]
        edges = [EdgeFactory(result=result, from_node=nodes[i], to_node=nodes[j]) for i, j in [(0, 1), (1, 2)]]
        sepsets = [SepsetFactory(result=result, from_node=nodes[0], to_node=nodes[2])]

        # When
        full_result = self.get(self.url_for(ResultResource, result_id=result.id))

        # Then
        assert full_result['id'] == result.id

        node_ids = {n.id for n in nodes}
        for node in full_result['nodes']:
            assert node['id'] in node_ids
            node_ids.remove(node['id'])
        assert len(node_ids) == 0

        edge_ids = {e.id for e in edges}
        for edge in full_result['edges']:
            assert edge['id'] in edge_ids
            edge_ids.remove(edge['id'])
        assert len(edge_ids) == 0

        sepset_ids = {s.id for s in sepsets}
        for sepset in full_result['sepsets']:
            assert sepset['id'] in sepset_ids
            sepset_ids.remove(sepset['id'])
        assert len(sepset_ids) == 0

    def test_result_compare(self):
        result1 = ResultFactory()
        result2 = ResultFactory()
        resp = self.get(self.url_for(ResultCompareResource, result_id=result1.id, other_result_id=result2.id))
        assert 'mean_jaccard_coefficient' in resp
        assert 'graph_edit_distance' in resp
        assert 'error_types' in resp
        assert 'hamming_distance' in resp
        assert 'hamming_distance_pcdag' in resp

    def test_delete_result(self):
        result = ResultFactory()
        nodes = [NodeFactory(dataset=result.job.experiment.dataset) for _ in range(3)]

        for i in range(3):
            EdgeFactory(from_node=nodes[i], to_node=nodes[(i + 1) % 3], result=result)

        assert len(db.session.query(Result).all()) == 1
        assert len(db.session.query(Node).all()) == 3
        assert len(db.session.query(Edge).all()) == 3

        deleted_result = self.delete(self.url_for(ResultResource,
                                                  result_id=result.id))

        assert deleted_result['id'] == result.id
        assert len(db.session.query(Result).all()) == 0
        assert len(db.session.query(Node).all()) == 3
        assert len(db.session.query(Edge).all()) == 0
