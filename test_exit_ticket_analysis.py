import pytest
import asyncio
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket
from unittest.mock import patch, MagicMock, AsyncMock
import json
from datetime import datetime, timedelta

from app.main import app
from app.models import ExitTicketClasswideAnalysis, Assignment, Class, CustomUser
from app.db import SessionLocal

# Create test client
client = TestClient(app)


class TestExitTicketAnalysisWebSocket:
    """Test the WebSocket endpoint for exit ticket analysis chat"""
    
    def setup_method(self):
        """Set up test data"""
        self.db = SessionLocal()
        
        # Create test teacher
        self.teacher = CustomUser(
            username='teacher@test.com',
            email='teacher@test.com',
            first_name='Test',
            last_name='Teacher',
            role='teacher'
        )
        self.db.add(self.teacher)
        self.db.commit()
        
        # Create test class
        self.test_class = Class(
            name='Test Class',
            teacher_id=self.teacher.id,
            period=1,
            subject='English'
        )
        self.db.add(self.test_class)
        self.db.commit()
        
        # Create test assignment
        self.assignment = Assignment(
            title='Test Exit Ticket',
            description='Test description',
            class_instance_id=self.test_class.id,
            due_date=datetime.now() + timedelta(hours=1),
            type='exit_ticket'
        )
        self.db.add(self.assignment)
        self.db.commit()
        
        # Create test analysis
        self.analysis = ExitTicketClasswideAnalysis(
            assignment_id=self.assignment.id,
            mastery_percentage=85.0,
            class_readiness='Ready (75%+ mastery)',
            most_common_misconceptions=['Test misconception'],
            error_severity_minor=1,
            error_severity_moderate=0,
            error_severity_major=0,
            conceptual_errors=0,
            procedural_errors=1,
            both_error_types=0,
            similar_intervention_needs=[],
            priority_intervention_areas=[],
            reteaching_focus='Test focus',
            clarity_issues='',
            success_indicators='Good evidence use',
            student_groupings_by_misconception=[],
            total_responses=1,
            completion_rate=100.0,
            thread_id='thread_test_123'
        )
        self.db.add(self.analysis)
        self.db.commit()
    
    def teardown_method(self):
        """Clean up test data"""
        self.db.close()
    
    @patch('app.main.os.getenv')
    def test_websocket_connection_success(self, mock_getenv):
        """Test successful WebSocket connection"""
        mock_getenv.return_value = 'test_assistant_id'
        
        with client.websocket_connect(f"/ws/exit-ticket-analysis/{self.analysis.thread_id}") as websocket:
            # Should receive initial connection message
            data = websocket.receive_json()
            assert data["type"] == "init"
            assert "Ready to discuss" in data["data"]["message"]
            assert data["data"]["thread_id"] == self.analysis.thread_id
            assert data["data"]["assistant_id"] == 'test_assistant_id'
    
    def test_websocket_connection_no_analysis(self):
        """Test WebSocket connection with invalid thread_id"""
        with client.websocket_connect("/ws/exit-ticket-analysis/invalid_thread") as websocket:
            # Should receive error message
            data = websocket.receive_json()
            assert data["type"] == "error"
            assert "No analysis found" in data["data"]["message"]
    
    @patch('app.main.os.getenv')
    def test_websocket_connection_no_assistant_id(self, mock_getenv):
        """Test WebSocket connection when assistant ID is not configured"""
        mock_getenv.return_value = None
        
        with client.websocket_connect(f"/ws/exit-ticket-analysis/{self.analysis.thread_id}") as websocket:
            # Should receive error message
            data = websocket.receive_json()
            assert data["type"] == "error"
            assert "EXIT_TICKET_CLASS_ID assistant not configured" in data["data"]["message"]
    
    @patch('app.main.os.getenv')
    @patch('app.main.client.beta.threads.messages.create')
    @patch('app.main.client.beta.threads.runs.create')
    def test_websocket_init_message(self, mock_run_create, mock_message_create, mock_getenv):
        """Test handling of init message type"""
        mock_getenv.return_value = 'test_assistant_id'
        mock_message_create.return_value = AsyncMock()
        
        # Mock streaming response
        mock_event = MagicMock()
        mock_event.event = "thread.message.delta"
        mock_event.data.delta.content = [MagicMock()]
        mock_event.data.delta.content[0].type = "text"
        mock_event.data.delta.content[0].text.value = "Hello"
        
        mock_run_create.return_value = AsyncMock()
        mock_run_create.return_value.__aiter__ = AsyncMock(return_value=iter([mock_event]))
        
        with client.websocket_connect(f"/ws/exit-ticket-analysis/{self.analysis.thread_id}") as websocket:
            # Skip initial connection message
            websocket.receive_json()
            
            # Send init message
            websocket.send_json({
                "type": "init",
                "message": "You are a helpful teacher's assistant..."
            })
            
            # Should receive token response
            data = websocket.receive_json()
            assert data["type"] == "token"
            assert data["data"]["content"] == "Hello"
    
    @patch('app.main.os.getenv')
    @patch('app.main.client.beta.threads.messages.create')
    @patch('app.main.client.beta.threads.runs.create')
    def test_websocket_regular_message(self, mock_run_create, mock_message_create, mock_getenv):
        """Test handling of regular message type"""
        mock_getenv.return_value = 'test_assistant_id'
        mock_message_create.return_value = AsyncMock()
        
        # Mock streaming response
        mock_event = MagicMock()
        mock_event.event = "thread.run.completed"
        
        mock_run_create.return_value = AsyncMock()
        mock_run_create.return_value.__aiter__ = AsyncMock(return_value=iter([mock_event]))
        
        with client.websocket_connect(f"/ws/exit-ticket-analysis/{self.analysis.thread_id}") as websocket:
            # Skip initial connection message
            websocket.receive_json()
            
            # Send regular message
            websocket.send_json({
                "type": "message",
                "message": "What does the analysis show?"
            })
            
            # Should receive completion response
            data = websocket.receive_json()
            assert data["type"] == "message"


class TestExitTicketAnalysisAPI:
    """Test the FastAPI endpoint for exit ticket analysis"""
    
    def setup_method(self):
        """Set up test data"""
        self.db = SessionLocal()
        
        # Create test teacher
        self.teacher = CustomUser(
            username='teacher@test.com',
            email='teacher@test.com',
            first_name='Test',
            last_name='Teacher',
            role='teacher'
        )
        self.db.add(self.teacher)
        self.db.commit()
        
        # Create test class
        self.test_class = Class(
            name='Test Class',
            teacher_id=self.teacher.id,
            period=1,
            subject='English'
        )
        self.db.add(self.test_class)
        self.db.commit()
        
        # Create test assignment
        self.assignment = Assignment(
            title='Test Exit Ticket',
            description='Test description',
            class_instance_id=self.test_class.id,
            due_date=datetime.now() + timedelta(hours=1),
            type='exit_ticket'
        )
        self.db.add(self.assignment)
        self.db.commit()
    
    def teardown_method(self):
        """Clean up test data"""
        self.db.close()
    
    @patch('app.main.os.getenv')
    @patch('app.main.client.beta.threads.create')
    @patch('app.main.client.beta.threads.messages.create')
    @patch('app.main.client.beta.threads.runs.create')
    def test_exit_ticket_classwide_api_success(self, mock_run_create, mock_message_create, mock_thread_create, mock_getenv):
        """Test successful exit ticket classwide analysis"""
        mock_getenv.return_value = 'test_assistant_id'
        
        # Mock thread creation
        mock_thread = MagicMock()
        mock_thread.id = 'thread_test_123'
        mock_thread_create.return_value = mock_thread
        
        # Mock message creation
        mock_message_create.return_value = AsyncMock()
        
        # Mock run creation and tool call
        mock_run = MagicMock()
        mock_run.id = 'run_test_123'
        mock_run.status = 'requires_action'
        mock_run.required_action.submit_tool_outputs.tool_calls = [
            MagicMock(
                id='tool_call_123',
                function=MagicMock(
                    name='analyze_classwide_exit_tickets',
                    arguments=json.dumps({
                        'mastery_percentage': 85.0,
                        'class_readiness': 'Ready (75%+ mastery)',
                        'most_common_misconceptions': ['Test misconception'],
                        'error_severity_distribution': {
                            'minor': 1,
                            'moderate': 0,
                            'major': 0
                        },
                        'conceptual_vs_procedural_breakdown': {
                            'conceptual': 0,
                            'procedural': 1,
                            'both': 0
                        },
                        'similar_intervention_needs': [],
                        'priority_intervention_areas': [],
                        'reteaching_focus': 'Test focus',
                        'clarity_issues': '',
                        'success_indicators': 'Good evidence use',
                        'student_groupings_by_misconception': []
                    })
                )
            )
        ]
        
        mock_run_create.return_value = mock_run
        
        # Test the API endpoint
        response = client.post("/exit_ticket_classwide_api/", json={
            "assignment_id": self.assignment.id
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["thread_id"] == 'thread_test_123'
        assert data["analysis"]["mastery_percentage"] == 85.0
        assert data["analysis"]["class_readiness"] == 'Ready (75%+ mastery)'
    
    def test_exit_ticket_classwide_api_not_found(self):
        """Test API with non-existent assignment"""
        response = client.post("/exit_ticket_classwide_api/", json={
            "assignment_id": 99999
        })
        
        assert response.status_code == 404
        data = response.json()
        assert "Exit ticket not found" in data["detail"]
    
    @patch('app.main.os.getenv')
    def test_exit_ticket_classwide_api_no_assistant_id(self, mock_getenv):
        """Test API when assistant ID is not configured"""
        mock_getenv.return_value = None
        
        response = client.post("/exit_ticket_classwide_api/", json={
            "assignment_id": self.assignment.id
        })
        
        assert response.status_code == 500
        data = response.json()
        assert "EXIT_TICKET_CLASS_ID assistant not configured" in data["detail"]


class TestExitTicketAnalysisIntegration:
    """Integration tests for the complete exit ticket analysis flow"""
    
    def setup_method(self):
        """Set up test data"""
        self.db = SessionLocal()
        
        # Create complete test scenario
        self.teacher = CustomUser(
            username='teacher@test.com',
            email='teacher@test.com',
            first_name='Test',
            last_name='Teacher',
            role='teacher'
        )
        self.db.add(self.teacher)
        self.db.commit()
        
        self.test_class = Class(
            name='Test Class',
            teacher_id=self.teacher.id,
            period=1,
            subject='English'
        )
        self.db.add(self.test_class)
        self.db.commit()
        
        self.assignment = Assignment(
            title='Test Exit Ticket',
            description='Test description',
            class_instance_id=self.test_class.id,
            due_date=datetime.now() + timedelta(hours=1),
            type='exit_ticket'
        )
        self.db.add(self.assignment)
        self.db.commit()
    
    def teardown_method(self):
        """Clean up test data"""
        self.db.close()
    
    @patch('app.main.os.getenv')
    @patch('app.main.client.beta.threads.create')
    @patch('app.main.client.beta.threads.messages.create')
    @patch('app.main.client.beta.threads.runs.create')
    def test_complete_analysis_to_chat_flow(self, mock_run_create, mock_message_create, mock_thread_create, mock_getenv):
        """Test complete flow from analysis creation to chat interaction"""
        mock_getenv.return_value = 'test_assistant_id'
        
        # Step 1: Create analysis via API
        mock_thread = MagicMock()
        mock_thread.id = 'thread_integration_test'
        mock_thread_create.return_value = mock_thread
        
        mock_message_create.return_value = AsyncMock()
        
        mock_run = MagicMock()
        mock_run.id = 'run_test_123'
        mock_run.status = 'requires_action'
        mock_run.required_action.submit_tool_outputs.tool_calls = [
            MagicMock(
                id='tool_call_123',
                function=MagicMock(
                    name='analyze_classwide_exit_tickets',
                    arguments=json.dumps({
                        'mastery_percentage': 90.0,
                        'class_readiness': 'Ready (75%+ mastery)',
                        'most_common_misconceptions': [],
                        'error_severity_distribution': {
                            'minor': 0,
                            'moderate': 0,
                            'major': 0
                        },
                        'conceptual_vs_procedural_breakdown': {
                            'conceptual': 0,
                            'procedural': 0,
                            'both': 0
                        },
                        'similar_intervention_needs': [],
                        'priority_intervention_areas': [],
                        'reteaching_focus': 'Great work overall',
                        'clarity_issues': '',
                        'success_indicators': 'Excellent evidence use',
                        'student_groupings_by_misconception': []
                    })
                )
            )
        ]
        
        mock_run_create.return_value = mock_run
        
        # Create analysis via API
        response = client.post("/exit_ticket_classwide_api/", json={
            "assignment_id": self.assignment.id
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["thread_id"] == 'thread_integration_test'
        
        # Step 2: Verify analysis was created in database
        analysis = self.db.query(ExitTicketClasswideAnalysis).filter(
            ExitTicketClasswideAnalysis.thread_id == 'thread_integration_test'
        ).first()
        
        assert analysis is not None
        assert analysis.mastery_percentage == 90.0
        assert analysis.thread_id == 'thread_integration_test'
        
        # Step 3: Test WebSocket connection using the thread_id
        with client.websocket_connect(f"/ws/exit-ticket-analysis/{analysis.thread_id}") as websocket:
            # Should receive initial connection message
            init_data = websocket.receive_json()
            assert init_data["type"] == "init"
            assert analysis.assignment.title in init_data["data"]["message"]
            assert init_data["data"]["thread_id"] == analysis.thread_id
            
            # This verifies the complete flow from analysis creation to chat functionality
            print(f"✅ Integration test passed: Analysis created with thread_id {analysis.thread_id} and chat connection established")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])