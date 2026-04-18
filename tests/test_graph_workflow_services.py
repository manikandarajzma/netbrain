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
    @patch("services.network_ops_workflow_service.pending_approval_store.get", return_value=None)
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
        _mock_get_pending_approval,
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

    @patch("services.network_ops_workflow_service.execute_pending_write_action", new_callable=AsyncMock, return_value="Created ServiceNow change request CHG0030042.\nstate: New")
    @patch("services.network_ops_workflow_service.response_presenter.build_network_ops_content")
    @patch("services.network_ops_workflow_service.pending_approval_store.clear")
    @patch(
        "services.network_ops_workflow_service.pending_approval_store.get",
        return_value={
            "approval_id": "approval-1",
            "tool_name": "create_servicenow_change_request",
            "action_label": "Create ServiceNow change request",
            "fields": {"short_description": "route map update"},
            "original_prompt": "create a change request for arista-ai1 route map update",
        },
    )
    @patch("services.network_ops_workflow_service.status_service.push", new_callable=AsyncMock)
    @patch("services.network_ops_workflow_service.nornir_client.clear_session_cache")
    @patch("services.network_ops_workflow_service.session_store.pop")
    @patch("services.network_ops_workflow_service.build_agent")
    async def test_network_ops_workflow_rejects_stale_structured_confirm(
        self,
        mock_build_agent,
        mock_session_pop,
        _mock_clear_cache,
        _mock_status_push,
        _mock_get_pending_approval,
        mock_clear_pending_approval,
        mock_presenter,
        mock_execute_pending,
    ):
        mock_session_pop.return_value = {}
        mock_presenter.return_value = {"direct_answer": "That approval request is no longer current. Use the latest approval card."}

        result = await network_ops_workflow_service.run(
            {
                "prompt": "Confirm",
                "session_id": "s-netops-confirm",
                "ui_action": {"type": "confirm", "approval_id": "wrong-id"},
            }
        )

        self.assertEqual(
            result["final_response"]["content"]["direct_answer"],
            "That approval request is no longer current. Use the latest approval card.",
        )
        mock_execute_pending.assert_not_awaited()
        mock_clear_pending_approval.assert_not_called()
        mock_build_agent.assert_not_called()

    @patch("services.network_ops_workflow_service.execute_pending_write_action", new_callable=AsyncMock, return_value="Created ServiceNow change request CHG0030042.\nstate: New")
    @patch("services.network_ops_workflow_service.response_presenter.build_network_ops_content")
    @patch("services.network_ops_workflow_service.pending_approval_store.clear")
    @patch(
        "services.network_ops_workflow_service.pending_approval_store.get",
        return_value={
            "approval_id": "approval-1",
            "tool_name": "create_servicenow_change_request",
            "action_label": "Create ServiceNow change request",
            "fields": {"short_description": "route map update"},
            "original_prompt": "create a change request for arista-ai1 route map update",
        },
    )
    @patch("services.network_ops_workflow_service.status_service.push", new_callable=AsyncMock)
    @patch("services.network_ops_workflow_service.nornir_client.clear_session_cache")
    @patch("services.network_ops_workflow_service.session_store.pop")
    @patch("services.network_ops_workflow_service.build_agent")
    async def test_network_ops_workflow_executes_pending_approval_on_structured_confirm(
        self,
        mock_build_agent,
        mock_session_pop,
        _mock_clear_cache,
        _mock_status_push,
        _mock_get_pending_approval,
        mock_clear_pending_approval,
        mock_presenter,
        mock_execute_pending,
    ):
        mock_session_pop.return_value = {}
        mock_presenter.return_value = {"direct_answer": "Created ServiceNow change request CHG0030042"}

        result = await network_ops_workflow_service.run(
            {
                "prompt": "Confirm",
                "session_id": "s-netops-confirm",
                "ui_action": {"type": "confirm", "approval_id": "approval-1"},
            }
        )

        self.assertEqual(result["final_response"]["content"]["direct_answer"], "Created ServiceNow change request CHG0030042")
        mock_execute_pending.assert_awaited_once()
        mock_clear_pending_approval.assert_called_once_with("s-netops-confirm")
        mock_build_agent.assert_not_called()

    @patch("services.network_ops_workflow_service.response_presenter.build_network_ops_content")
    @patch("services.network_ops_workflow_service.pending_approval_store.clear")
    @patch(
        "services.network_ops_workflow_service.pending_approval_store.get",
        return_value={
            "tool_name": "create_servicenow_incident",
            "action_label": "Create ServiceNow incident",
            "fields": {"short_description": "Connectivity issue"},
            "original_prompt": "create an incident for connectivity issue",
        },
    )
    @patch("services.network_ops_workflow_service.status_service.push", new_callable=AsyncMock)
    @patch("services.network_ops_workflow_service.nornir_client.clear_session_cache")
    @patch("services.network_ops_workflow_service.session_store.pop")
    @patch("services.network_ops_workflow_service.build_agent")
    async def test_network_ops_workflow_cancels_pending_approval(
        self,
        mock_build_agent,
        mock_session_pop,
        _mock_clear_cache,
        _mock_status_push,
        _mock_get_pending_approval,
        mock_clear_pending_approval,
        mock_presenter,
    ):
        mock_session_pop.return_value = {}
        mock_presenter.return_value = {"direct_answer": "Cancelled"}

        result = await network_ops_workflow_service.run(
            {
                "prompt": "Cancel",
                "session_id": "s-netops-cancel",
                "ui_action": {"type": "cancel"},
            }
        )

        self.assertEqual(result["final_response"]["content"]["direct_answer"], "Cancelled")
        mock_clear_pending_approval.assert_called_once_with("s-netops-cancel")
        mock_build_agent.assert_not_called()

    @patch("services.network_ops_workflow_service.metrics_recorder.observe_ms")
    @patch("services.network_ops_workflow_service.metrics_recorder.increment")
    @patch("services.network_ops_workflow_service.log_event")
    @patch("services.network_ops_workflow_service.response_presenter.build_network_ops_content")
    @patch("services.network_ops_workflow_service.extract_final_text", return_value="Proposed action: Create ServiceNow change request")
    @patch("services.network_ops_workflow_service.network_ops_scenario_service.select_scenario", new_callable=AsyncMock, return_value="change_record")
    @patch("services.network_ops_workflow_service.pending_approval_store.clear")
    @patch(
        "services.network_ops_workflow_service.pending_approval_store.get",
        side_effect=[
            {
                "tool_name": "create_servicenow_change_request",
                "action_label": "Create ServiceNow change request",
                "fields": {"short_description": "route map update", "ci_name": "arista-ai1"},
                "original_prompt": "create a change request for arista-ai1 route map update",
            },
            None,
            None,
        ],
    )
    @patch("services.network_ops_workflow_service.status_service.push", new_callable=AsyncMock)
    @patch("services.network_ops_workflow_service.nornir_client.clear_session_cache")
    @patch("services.network_ops_workflow_service.session_store.pop")
    @patch("services.network_ops_workflow_service.memory_manager.clear_pending_context")
    @patch("services.network_ops_workflow_service.memory_manager.get_pending_context", return_value=(None, None))
    @patch("services.network_ops_workflow_service.build_agent")
    async def test_network_ops_workflow_rewrites_pending_approval_for_edits(
        self,
        mock_build_agent,
        _mock_get_pending_context,
        _mock_clear_pending_context,
        mock_session_pop,
        _mock_clear_cache,
        _mock_status_push,
        _mock_get_pending_approval,
        mock_clear_pending_approval,
        _mock_select_scenario,
        _mock_extract_final_text,
        mock_presenter,
        _mock_log_event,
        _mock_increment,
        _mock_observe_ms,
    ):
        mock_session_pop.side_effect = [{}, {}]
        mock_presenter.return_value = {"direct_answer": "updated proposal"}
        agent = AsyncMock()
        agent.ainvoke.return_value = {"messages": ["ignored"]}
        mock_build_agent.return_value = agent

        await network_ops_workflow_service.run(
            {
                "prompt": "change the justification to restore routing for business x",
                "session_id": "s-netops-edit",
            }
        )

        forwarded_prompt = mock_build_agent.call_args.args[0]
        self.assertIn("pending ServiceNow write action", forwarded_prompt)
        self.assertIn("route map update", forwarded_prompt)
        self.assertIn("restore routing for business x", forwarded_prompt)
        mock_clear_pending_approval.assert_called_once_with("s-netops-edit")


if __name__ == "__main__":
    unittest.main()
