from src.master.resources import NodeResource, ResultNodeListResource, NodeContextResource
from test.factories import DatasetFactory, NodeFactory, EdgeFactory, ResultFactory
from .base import BaseResourceTest


class NodeTest(BaseResourceTest):
    def test_returns_all_nodes_for_result(self):
        # Given
        result = ResultFactory()
        nodes = [NodeFactory(dataset=result.job.experiment.dataset) for _ in range(3)]

        # When
        results = self.get(self.url_for(ResultNodeListResource, result_id=result.id))

        # Then
        assert len(results) == len(nodes)
        node_ids = {n.id for n in nodes}
        for node in results:
            assert node['dataset_id'] == result.job.experiment.dataset_id
            assert node['id'] in node_ids
            node_ids.remove(node['id'])
        assert len(node_ids) == 0

    def test_returns_my_node(self):
        # Given
        node = NodeFactory()

        # When
        result = self.get(self.url_for(NodeResource, node_id=node.id))

        # Then
        assert result['id'] == node.id
        assert result['dataset_id'] == node.dataset_id

    def test_returns_node_context(self):
        # Given
        result = ResultFactory()
        nodes = [NodeFactory(dataset=result.job.experiment.dataset) for _ in range(3)]
        edges = [EdgeFactory(result=result, from_node=nodes[i], to_node=nodes[j]) for i, j in [(0, 1), (1, 2), (0, 2)]]
        main_node = nodes[1]

        # When
        context = self.get(self.url_for(NodeContextResource, node_id=main_node.id))

        assert context['main_node']['id'] == main_node.id

        context_node_ids = {n.id for n in nodes if n != main_node}
        for context_node in context['context_nodes']:
            assert context_node['dataset_id'] == result.job.experiment.dataset_id
            assert context_node['id'] in context_node_ids
            context_node_ids.remove(context_node['id'])
        assert len(context_node_ids) == 0

        edge_ids = {edges[0].id, edges[1].id}  # Only edges connected to main_node
        for edge in context['edges']:
            assert edge['result_id'] == result.id
            assert edge['id'] in edge_ids
            edge_ids.remove(edge['id'])
        assert len(edge_ids) == 0
