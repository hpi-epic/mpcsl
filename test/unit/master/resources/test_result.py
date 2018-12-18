from src.db import db
from src.master.resources.results import ResultListResource, ResultResource
from src.models import Node, Result
from test.factories import ResultFactory, NodeFactory, EdgeFactory, SepsetFactory
from .base import BaseResourceTest


class JobTest(BaseResourceTest):
    def test_returns_all_results(self):
        # Given
        result = ResultFactory()
        result2 = ResultFactory()

        # When
        results = self.get(self.api.url_for(ResultListResource))

        # Then
        assert len(results) == 2
        assert results[0]['id'] == result.id
        assert results[1]['id'] == result2.id

    def test_returns_my_job(self):
        # Given
        result = ResultFactory()
        nodes = [NodeFactory(result=result) for _ in range(3)]
        edges = [EdgeFactory(result=result, from_node=nodes[i], to_node=nodes[j]) for i, j in [(0, 1), (1, 2)]]
        sepsets = [SepsetFactory(result=result, from_node=nodes[0], to_node=nodes[2], node_names=[nodes[1].name])]

        # When
        full_result = self.get(self.api.url_for(ResultResource, result_id=result.id))

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

    def test_delete_job(self):
        # Given
        result = ResultFactory()
        for _ in range(3):
            NodeFactory(result=result)

        # When
        deleted_result = self.delete(self.api.url_for(ResultResource, result_id=result.id))

        # Then
        assert deleted_result['id'] == result.id
        assert len(db.session.query(Result).all()) == 0
        assert len(db.session.query(Node).all()) == 0
