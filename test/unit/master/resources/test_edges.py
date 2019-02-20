from src.master.resources import EdgeResource, ResultEdgeListResource
from test.factories import ResultFactory, NodeFactory, EdgeFactory
from .base import BaseResourceTest


class EdgeTest(BaseResourceTest):
    def test_returns_all_edges_for_result(self):
        # Given
        result = ResultFactory()
        nodes = [NodeFactory(dataset=result.job.experiment.dataset) for _ in range(3)]
        edges = [EdgeFactory(result=result, from_node=nodes[i], to_node=nodes[j]) for i, j in [(0, 1), (1, 2)]]

        # When
        results = self.get(self.url_for(ResultEdgeListResource, result_id=result.id))

        # Then
        assert len(results) == len(edges)
        edge_ids = {e.id for e in edges}
        for edge in results:
            assert edge['result_id'] == result.id
            assert edge['id'] in edge_ids
            edge_ids.remove(edge['id'])
        assert len(edge_ids) == 0

    def test_returns_my_edge(self):
        # Given
        edge = EdgeFactory()

        # When
        result = self.get(self.url_for(EdgeResource, edge_id=edge.id))

        # Then
        assert result['id'] == edge.id
        assert result['result_id'] == edge.result_id
