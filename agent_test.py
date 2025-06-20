"""
Test script for the Legal Document AI Agent
Validates functionality before OpenWebUI deployment
"""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import asyncio
import os
import logging
from datetime import datetime
from typing import Dict, Any

# Configure logging for testing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from ai_agents.legal_document_agent import legal_document_agent, LegalResponse
from config.agent_settings import agent_settings, validate_agent_configuration
from utils.logger import setup_logging

class AgentTester:
    """Test harness for the Legal Document AI Agent"""
    
    def __init__(self):
        self.agent = legal_document_agent
        self.test_user_id = "test_user_001"
        self.test_results = []
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        print("🧪 LEGAL DOCUMENT AI AGENT TEST SUITE")
        print("=" * 60)
        
        results = {
            "start_time": datetime.now().isoformat(),
            "tests": {},
            "overall_status": "PENDING",
            "errors": [],
            "warnings": []
        }
        
        # Test 1: Configuration Validation
        print("\n1️⃣ Testing Configuration...")
        try:
            config_result = await self.test_configuration()
            results["tests"]["configuration"] = config_result
            if config_result["status"] == "PASS":
                print("   ✅ Configuration test passed")
            else:
                print("   ❌ Configuration test failed")
                results["errors"].extend(config_result.get("errors", []))
        except Exception as e:
            print(f"   💥 Configuration test crashed: {str(e)}")
            results["tests"]["configuration"] = {"status": "ERROR", "error": str(e)}
            results["errors"].append(f"Configuration test: {str(e)}")
        
        # Test 2: Health Check
        print("\n2️⃣ Testing Health Status...")
        try:
            health_result = await self.test_health_status()
            results["tests"]["health"] = health_result
            if health_result["status"] == "PASS":
                print("   ✅ Health check passed")
            else:
                print("   ❌ Health check failed")
                results["errors"].extend(health_result.get("errors", []))
        except Exception as e:
            print(f"   💥 Health check crashed: {str(e)}")
            results["tests"]["health"] = {"status": "ERROR", "error": str(e)}
            results["errors"].append(f"Health check: {str(e)}")
        
        # Test 3: Basic Query
        print("\n3️⃣ Testing Basic Document Query...")
        try:
            query_result = await self.test_basic_query()
            results["tests"]["basic_query"] = query_result
            if query_result["status"] == "PASS":
                print("   ✅ Basic query test passed")
            else:
                print("   ❌ Basic query test failed")
                results["errors"].extend(query_result.get("errors", []))
        except Exception as e:
            print(f"   💥 Basic query test crashed: {str(e)}")
            results["tests"]["basic_query"] = {"status": "ERROR", "error": str(e)}
            results["errors"].append(f"Basic query: {str(e)}")
        
        # Test 4: Case Summary
        print("\n4️⃣ Testing Case Summary...")
        try:
            summary_result = await self.test_case_summary()
            results["tests"]["case_summary"] = summary_result
            if summary_result["status"] == "PASS":
                print("   ✅ Case summary test passed")
            else:
                print("   ❌ Case summary test failed")
                results["errors"].extend(summary_result.get("errors", []))
        except Exception as e:
            print(f"   💥 Case summary test crashed: {str(e)}")
            results["tests"]["case_summary"] = {"status": "ERROR", "error": str(e)}
            results["errors"].append(f"Case summary: {str(e)}")
        
        # Test 5: Security Validation
        print("\n5️⃣ Testing Security & Case Isolation...")
        try:
            security_result = await self.test_security()
            results["tests"]["security"] = security_result
            if security_result["status"] == "PASS":
                print("   ✅ Security test passed")
            else:
                print("   ❌ Security test failed")
                results["errors"].extend(security_result.get("errors", []))
        except Exception as e:
            print(f"   💥 Security test crashed: {str(e)}")
            results["tests"]["security"] = {"status": "ERROR", "error": str(e)}
            results["errors"].append(f"Security test: {str(e)}")
        
        # Determine overall status
        test_statuses = [test.get("status", "ERROR") for test in results["tests"].values()]
        if all(status == "PASS" for status in test_statuses):
            results["overall_status"] = "PASS"
        elif "ERROR" in test_statuses:
            results["overall_status"] = "ERROR"
        else:
            results["overall_status"] = "FAIL"
        
        results["end_time"] = datetime.now().isoformat()
        
        # Print summary
        self.print_test_summary(results)
        
        return results
    
    async def test_configuration(self) -> Dict[str, Any]:
        """Test configuration validity"""
        try:
            validation = validate_agent_configuration()
            if validation["valid"]:
                return {
                    "status": "PASS",
                    "details": "Configuration validation successful",
                    "warnings": validation.get("warnings", [])
                }
            else:
                return {
                    "status": "FAIL",
                    "details": "Configuration validation failed",
                    "errors": validation.get("errors", [])
                }
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e)
            }
    
    async def test_health_status(self) -> Dict[str, Any]:
        """Test agent health status"""
        try:
            health = self.agent.get_health_status()
            
            if health["status"] == "healthy":
                return {
                    "status": "PASS",
                    "details": "Agent health check successful",
                    "case_accessible": health.get("case_accessible", False),
                    "vector_store_healthy": health.get("vector_store_healthy", False)
                }
            else:
                return {
                    "status": "FAIL",
                    "details": f"Agent unhealthy: {health.get('error', 'Unknown error')}",
                    "errors": [health.get("error", "Health check failed")]
                }
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e)
            }
    
    async def test_basic_query(self) -> Dict[str, Any]:
        """Test basic document querying"""
        try:
            test_query = "What is this case about? Provide a brief overview."
            
            response = await self.agent.query_documents(
                user_query=test_query,
                user_id=self.test_user_id
            )
            
            # Validate response structure
            if not isinstance(response, LegalResponse):
                return {
                    "status": "FAIL",
                    "errors": ["Response is not a LegalResponse object"]
                }
            
            if not response.answer:
                return {
                    "status": "FAIL",
                    "errors": ["Empty answer in response"]
                }
            
            if response.confidence < 0 or response.confidence > 1:
                return {
                    "status": "FAIL",
                    "errors": ["Invalid confidence score"]
                }
            
            return {
                "status": "PASS",
                "details": "Basic query successful",
                "response_length": len(response.answer),
                "confidence": response.confidence,
                "sources_count": len(response.sources),
                "query": test_query
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e)
            }
    
    async def test_case_summary(self) -> Dict[str, Any]:
        """Test case summary generation"""
        try:
            response = await self.agent.get_case_summary(self.test_user_id)
            
            if not isinstance(response, LegalResponse):
                return {
                    "status": "FAIL",
                    "errors": ["Response is not a LegalResponse object"]
                }
            
            if not response.answer:
                return {
                    "status": "FAIL",
                    "errors": ["Empty case summary"]
                }
            
            return {
                "status": "PASS",
                "details": "Case summary generated successfully",
                "summary_length": len(response.answer),
                "confidence": response.confidence,
                "sources_count": len(response.sources)
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e)
            }
    
    async def test_security(self) -> Dict[str, Any]:
        """Test security and case isolation"""
        try:
            # Test allowed case
            allowed_case = agent_settings.default_case_name
            
            # Test case access validation
            from utils.validators import validate_case_access, CaseAccessError
            
            # Should succeed
            try:
                validate_case_access(allowed_case)
                case_access_ok = True
            except CaseAccessError:
                case_access_ok = False
            
            # Should fail for wrong case
            try:
                validate_case_access("Wrong Case v. Invalid")
                wrong_case_blocked = False  # Should have thrown exception
            except CaseAccessError:
                wrong_case_blocked = True  # Correctly blocked
            
            if case_access_ok and wrong_case_blocked:
                return {
                    "status": "PASS",
                    "details": "Case isolation working correctly",
                    "allowed_case": allowed_case
                }
            else:
                errors = []
                if not case_access_ok:
                    errors.append("Valid case access failed")
                if not wrong_case_blocked:
                    errors.append("Invalid case access not blocked")
                
                return {
                    "status": "FAIL",
                    "errors": errors
                }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e)
            }
    
    def print_test_summary(self, results: Dict[str, Any]):
        """Print formatted test summary"""
        print("\n" + "=" * 60)
        print("🏁 TEST SUMMARY")
        print("=" * 60)
        
        # Overall status
        status_emoji = {
            "PASS": "✅",
            "FAIL": "❌", 
            "ERROR": "💥",
            "PENDING": "⏳"
        }
        
        print(f"Overall Status: {status_emoji[results['overall_status']]} {results['overall_status']}")
        
        # Individual test results
        print("\nTest Results:")
        for test_name, test_result in results["tests"].items():
            status = test_result.get("status", "UNKNOWN")
            emoji = status_emoji.get(status, "❓")
            print(f"  {emoji} {test_name.replace('_', ' ').title()}: {status}")
            
            if "details" in test_result:
                print(f"     {test_result['details']}")
        
        # Errors and warnings
        if results["errors"]:
            print(f"\n🚨 Errors ({len(results['errors'])}):")
            for error in results["errors"]:
                print(f"  - {error}")
        
        if results.get("warnings"):
            print(f"\n⚠️  Warnings ({len(results['warnings'])}):")
            for warning in results["warnings"]:
                print(f"  - {warning}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if results["overall_status"] == "PASS":
            print("  - Agent is ready for OpenWebUI deployment")
            print("  - Consider running performance tests under load")
            print("  - Monitor logs during initial deployment")
        elif results["overall_status"] == "FAIL":
            print("  - Fix configuration issues before deployment")
            print("  - Check vector store connectivity")
            print("  - Verify case data is properly loaded")
        else:
            print("  - Resolve critical errors before proceeding")
            print("  - Check environment variables and dependencies")
            print("  - Verify API keys and database connections")

async def main():
    """Main test execution"""
    print("🚀 Starting Legal Document AI Agent Tests...\n")
    
    tester = AgentTester()
    results = await tester.run_all_tests()
    
    # Exit with appropriate code
    if results["overall_status"] == "PASS":
        print("\n🎉 All tests passed! Agent is ready for deployment.")
        sys.exit(0)
    else:
        print(f"\n💔 Tests failed with status: {results['overall_status']}")
        print("Please fix the issues before deploying to OpenWebUI.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())