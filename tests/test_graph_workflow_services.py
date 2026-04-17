import unittest
from unittest.mock import AsyncMock, patch

from graph.graph_nodes import call_network_ops_agent, call_troubleshoot_agent
from services.network_ops_workflow_service import network_ops_workflow_service
from services.troubleshoot_workflow_service import troubleshoot_workflow_service


class GraphWorkflowDelegationTests(unittest.IsolatedAsyncioTestCase):
    @patch("graph.graph_nodes.troubleshoot_workflow_service.run", new_callable=AsyncMock)
    async def test_call_troubleshoot_agent_delegates_to_workflow_service(self, mock_run):
        mock_run.return_value = {"final_response": {"role": "assistant", "content": {"direct_answer": "ok"}}}
        state = {"prompt": "help me troubleshoot connectivity", "session_id": "s1"}

        result = await call_troubleshoot_agent(state)

        self.assertEqual(result["final_response"]["content"]["direct_answer"], "ok")
        mock_run.assert_awaited_once_with(state)

    @patch("graph.graph_nodes.network_ops_workflow_service.run", new_callable=AsyncMock)
    async def test_call_network_ops_agent_delegates_to_workflow_service(self, mock_run):
        mock_run.return_value = {"final_response": {"role": "assistant", "content": {"direct_answer": "ok"}}}
        state = {"prompt": "create an incident", "session_id": "s2"}

        result = await call_network_ops_agent(state)

        self.assertEqual(result["final_response"]["content"]["direct_answer"], "ok")
        mock_run.assert_awaited_once_with(state)


class TroubleshootWorkflowServiceTests(unittest.IsolatedAsyncioTestCase):
    @patch("services.troubleshoot_workflow_service.metrics_recorder.observe_ms")
    @patch("services.troubleshoot_workflow_service.metrics_recorder.increment")
    @patch("services.troubleshoot_workflow_service.log_event")
    @patch("services.troubleshoot_workflow_service.memory_manager.store_agent_memory_entry", new_callable=AsyncMock)
    @patch("services.troubleshoot_workflow_service.response_presenter.build_troubleshoot_content")
    @patch("services.troubleshoot_workflow_service.workflow_state_service.merge_session_data")
    @patch("services.troubleshoot_workflow_service.workflow_state_service.missing_path_visuals", return_value=False)
    @patch("services.troubleshoot_workflow_service.workflow_state_service.needs_connectivity_snapshot", return_value=False)
    @patch("services.troubleshoot_workflow_service.memory_manager.build_recall_follow_up", return_value="recall follow-up")
    @patch("services.troubleshoot_workflow_service.memory_manager.should_trigger_recall_follow_up", return_value=True)
    @patch("services.troubleshoot_workflow_service.extract_final_text", side_effect=["first answer", "second answer"])
    @patch("services.troubleshoot_workflow_service.resolve_incident_prompt", new_callable=AsyncMock)
    @patch("services.troubleshoot_workflow_service.troubleshoot_scenario_service.select_scenario", new_callable=AsyncMock, return_value="connectivity")
    @patch("services.troubleshoot_workflow_service.status_service.push", new_callable=AsyncMock)
    @patch("services.troubleshoot_workflow_service.nornir_client.clear_session_cache")
    @patch("services.troubleshoot_workflow_service.session_store.pop")
    @patch("services.troubleshoot_workflow_service.memory_manager.clear_pending_context")
    @patch("services.troubleshoot_workflow_service.memory_manager.get_pending_context", return_value=(None, None))
    @patch("services.troubleshoot_workflow_service.build_agent")
    async def test_troubleshoot_workflow_runs_evidence_driven_memory_follow_up(
        self,
        mock_build_agent,
        _mock_get_pending,
        _mock_clear_pending,
        mock_session_pop,
        _mock_clear_cache,
        _mock_status_push,
        _mock_select_scenario,
        mock_resolve_incident_prompt,
        _mock_extract_final_text,
        _mock_should_trigger,
        _mock_build_follow_up,
        _mock_needs_snapshot,
        _mock_missing_paths,
        mock_merge_session_data,
        mock_presenter,
        _mock_store_memory,
        _mock_log_event,
        _mock_increment,
        _mock_observe_ms,
    ):
        first_session = {
            "interface_details": {
                "arista-ai1:Ethernet1": {
                    "oper_status": "down",
                    "line_protocol": "down",
                }
            },
            "memory_recall_used": False,
        }
        second_session = {"memory_recall_used": True}
        merged_session = {
            "interface_details": first_session["interface_details"],
            "memory_recall_used": True,
        }
        mock_session_pop.side_effect = [{}, first_session, second_session]
        mock_merge_session_data.return_value = merged_session
        mock_presenter.return_value = {"direct_answer": "final answer"}
        mock_resolve_incident_prompt.return_value = (
            "help me troubleshoot connectivity from 10.0.100.100 to 10.0.200.200 on tcp port 443",
            None,
        )

        agent = AsyncMock()
        agent.ainvoke.side_effect = [
            {"messages": ["ignored-first-pass"]},
            {"messages": ["ignored-second-pass"]},
        ]
        mock_build_agent.return_value = agent

        result = await troubleshoot_workflow_service.run(
            {
                "prompt": "help me troubleshoot connectivity from 10.0.100.100 to 10.0.200.200 on tcp port 443",
                "session_id": "s-memory",
            }
        )

        self.assertEqual(result["final_response"]["content"]["direct_answer"], "final answer")
        self.assertEqual(agent.ainvoke.await_count, 2)
        mock_build_agent.assert_called_with(
            "help me troubleshoot connectivity from 10.0.100.100 to 10.0.200.200 on tcp port 443",
            "connectivity",
        )
        mock_presenter.assert_called_once_with(
            "second answer",
            merged_session,
            "help me troubleshoot connectivity from 10.0.100.100 to 10.0.200.200 on tcp port 443",
            None,
        )


class NetworkOpsWorkflowServiceTests(unittest.IsolatedAsyncioTestCase):
    @patch("services.network_ops_workflow_service.metrics_recorder.observe_ms")
    @patch("services.network_ops_workflow_service.metrics_recorder.increment")
    @patch("services.network_ops_workflow_service.log_event")
    @patch("services.network_ops_workflow_service.response_presenter.build_network_ops_content")
    @patch("services.network_ops_workflow_service.extract_final_text", return_value="created CHG0030042")
    @patch("services.network_ops_workflow_service.network_ops_scenario_service.select_scenario", new_callable=AsyncMock, return_value="change_record")
    @patch("services.network_ops_workflow_service.status_service.push", new_callable=AsyncMock)
    @patch("services.network_ops_workflow_service.nornir_client.clear_session_cache")
    @patch("services.network_ops_workflow_service.session_store.pop")
    @patch("services.network_ops_workflow_service.memory_manager.clear_pending_context")
    @patch("services.network_ops_workflow_service.memory_manager.get_pending_context", return_value=(None, None))
    @patch("services.network_ops_workflow_service.build_agent")
    async def test_network_ops_workflow_uses_llm_selected_scenario(
        self,
        mock_build_agent,
        _mock_get_pending,
        _mock_clear_pending,
        mock_session_pop,
        _mock_clear_cache,
        _mock_status_push,
        _mock_select_scenario,
        _mock_extract_final_text,
        mock_presenter,
        _mock_log_event,
        _mock_increment,
        _mock_observe_ms,
    ):
        mock_session_pop.side_effect = [{}, {}]
        mock_presenter.return_value = {"direct_answer": "done"}

        agent = AsyncMock()
        agent.ainvoke.return_value = {"messages": ["ignored"]}
        mock_build_agent.return_value = agent

        result = await network_ops_workflow_service.run(
            {
                "prompt": "create a change request for arista-ai1 route map update",
                "session_id": "s-netops",
            }
        )

        self.assertEqual(result["final_response"]["content"]["direct_answer"], "done")
        mock_build_agent.assert_called_once_with(
            "create a change request for arista-ai1 route map update",
            "change_record",
        )
        agent.ainvoke.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
