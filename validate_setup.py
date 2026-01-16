"""
RAG Starter Kit: Setup Validation

Validates that the environment is properly configured before running.
Provides clear error messages to help users diagnose issues.
"""

import os
from typing import List, Tuple, Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def validate_env_vars() -> Tuple[bool, List[str]]:
    """
    Validate that all required environment variables are set.
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    required_vars = {
        "SUPABASE_URL": "Get this from Supabase Dashboard → Settings → API → Project URL",
        "SUPABASE_SERVICE_KEY": "Get this from Supabase Dashboard → Settings → API → Service Role Key (not anon key!)",
        "OPENAI_API_KEY": "Get this from https://platform.openai.com/api-keys",
    }
    
    for var, help_text in required_vars.items():
        value = os.getenv(var)
        if not value:
            errors.append(f"❌ Missing {var}\n   💡 {help_text}")
        elif var == "SUPABASE_URL" and not value.startswith("https://"):
            errors.append(f"❌ Invalid SUPABASE_URL: Should start with 'https://'\n   💡 Current value: {value[:50]}...")
        elif var == "SUPABASE_SERVICE_KEY" and not value.startswith("eyJ"):
            errors.append(f"❌ Invalid SUPABASE_SERVICE_KEY: Should start with 'eyJ' (JWT token)\n   💡 You might be using the anon key instead of service role key")
        elif var == "OPENAI_API_KEY" and not value.startswith("sk-"):
            errors.append(f"❌ Invalid OPENAI_API_KEY: Should start with 'sk-'\n   💡 Current value: {value[:10]}...")
    
    return len(errors) == 0, errors


def validate_supabase_connection() -> Tuple[bool, Optional[str]]:
    """
    Validate that Supabase connection works and table exists with correct structure.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        from supabase import create_client
        
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        
        if not url or not key:
            return False, "Supabase credentials not set (run validate_env_vars first)"
        
        client = create_client(url, key)
        
        # Try to query the table
        result = client.table("site_pages").select("id").limit(1).execute()
        
        # Validate table structure by checking for required columns
        # We'll try to select all required columns to ensure they exist
        required_columns = ["id", "url", "chunk_number", "title", "summary", "content", "metadata", "embedding", "source"]
        try:
            test_result = client.table("site_pages").select(",".join(required_columns)).limit(1).execute()
        except Exception as col_error:
            error_msg = str(col_error).lower()
            if "column" in error_msg and "does not exist" in error_msg:
                return False, (
                    "❌ Table 'site_pages' is missing required columns\n"
                    "💡 See Step 1.2 Schema in the paid Substack guide to set up the database schema correctly."
                )
            # If it's a different error, continue (might be empty table)
        
        return True, None
        
    except Exception as e:
        error_msg = str(e)
        
        # Provide specific error messages
        if "relation" in error_msg.lower() and "does not exist" in error_msg.lower():
            return False, (
                "❌ Table 'site_pages' does not exist in Supabase\n"
                "💡 See Step 1.2 Schema in the paid Substack guide to set up the database schema."
            )
        elif "permission denied" in error_msg.lower() or "row-level security" in error_msg.lower():
            return False, (
                "❌ Permission denied accessing Supabase table\n"
                "💡 You might be using the anon key instead of service role key\n"
                "   Get the Service Role Key from: Settings → API → Service Role Key"
            )
        elif "invalid api key" in error_msg.lower() or "unauthorized" in error_msg.lower():
            return False, (
                "❌ Invalid Supabase credentials\n"
                "💡 Check your SUPABASE_URL and SUPABASE_SERVICE_KEY in .env file\n"
                "   Make sure you're using the Service Role Key (starts with 'eyJ'), not anon key"
            )
        elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            return False, (
                "❌ Cannot connect to Supabase\n"
                "💡 Check your internet connection and SUPABASE_URL\n"
                "   Verify your Supabase project is active (free tier pauses after inactivity)"
            )
        else:
            return False, f"❌ Supabase connection error: {error_msg}\n💡 Check your credentials and network connection"


def validate_supabase_function() -> Tuple[bool, Optional[str]]:
    """
    Validate that match_documents function exists.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        from supabase import create_client
        
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        client = create_client(url, key)
        
        # Try to call the function with a dummy embedding
        dummy_embedding = [0.0] * 1536
        result = client.rpc(
            "match_documents",
            {
                "query_embedding": dummy_embedding,
                "match_count": 1,
                "similarity_threshold": 0.0,
            }
        ).execute()
        
        return True, None
        
    except Exception as e:
        error_msg = str(e)
        
        if "function" in error_msg.lower() and "does not exist" in error_msg.lower():
            return False, (
                "❌ Function 'match_documents' does not exist\n"
                "💡 See Step 1.2 Schema in the paid Substack guide to set up the database schema."
            )
        elif "vector" in error_msg.lower() and "does not exist" in error_msg.lower():
            return False, (
                "❌ pgvector extension not enabled\n"
                "💡 See Step 1.2 Schema in the paid Substack guide to set up the database schema."
            )
        else:
            return False, (
                f"❌ Error calling match_documents: {error_msg}\n"
                "💡 See Step 1.2 Schema in the paid Substack guide to set up the database schema."
            )


def validate_openai_connection() -> Tuple[bool, Optional[str]]:
    """
    Validate that OpenAI API key works.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        from openai import OpenAI
        
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            return False, "OpenAI API key not set"
        
        client = OpenAI(api_key=key)
        
        # Try to list models (lightweight API call)
        models = client.models.list()
        
        return True, None
        
    except Exception as e:
        error_msg = str(e)
        
        if "invalid_api_key" in error_msg.lower() or "incorrect api key" in error_msg.lower():
            return False, (
                "❌ Invalid OpenAI API key\n"
                "💡 Get a new key from https://platform.openai.com/api-keys\n"
                "   Make sure the key starts with 'sk-'"
            )
        elif "insufficient_quota" in error_msg.lower() or "billing" in error_msg.lower():
            return False, (
                "❌ OpenAI account has insufficient quota or billing issue\n"
                "💡 Add a payment method at https://platform.openai.com/account/billing\n"
                "   Free tier gives $5 credit to start"
            )
        elif "rate_limit" in error_msg.lower():
            return False, (
                "❌ OpenAI rate limit exceeded\n"
                "💡 Wait a few minutes and try again\n"
                "   Consider upgrading your OpenAI plan for higher limits"
            )
        elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            return False, (
                "❌ Cannot connect to OpenAI API\n"
                "💡 Check your internet connection\n"
                "   OpenAI API might be experiencing issues (check status.openai.com)"
            )
        else:
            return False, f"❌ OpenAI API error: {error_msg}\n💡 Check your API key and account status"


def validate_data_directory() -> Tuple[bool, Optional[str], int]:
    """
    Validate that data directory exists and has files.
    
    Returns:
        (is_valid, error_message, file_count)
    """
    data_dir = Path("./data")
    
    if not data_dir.exists():
        return False, (
            "❌ Data directory './data' does not exist\n"
            "💡 Create it with: mkdir data\n"
            "   Then add .md files to it"
        ), 0
    
    if not data_dir.is_dir():
        return False, (
            "❌ './data' exists but is not a directory\n"
            "💡 Remove the file and create a directory instead"
        ), 0
    
    # Count markdown files
    md_files = list(data_dir.glob("*.md")) + list(data_dir.glob("*.markdown"))
    
    if len(md_files) == 0:
        return False, (
            "⚠️ No Markdown files found in ./data/\n"
            "💡 Add .md or .markdown files to the ./data/ directory"
        ), 0
    
    return True, None, len(md_files)


def validate_all() -> Tuple[bool, List[str], List[str]]:
    """
    Run all validation checks.
    
    Returns:
        (all_passed, list_of_errors, list_of_passed_tests)
    """
    errors = []
    passed_tests = []
    
    # Check environment variables
    print("🔍 Checking environment variables...")
    env_valid, env_errors = validate_env_vars()
    if env_valid:
        passed_tests.append("✅ Environment variables")
    else:
        errors.extend(env_errors)
        return False, errors, passed_tests  # Can't continue without env vars
    
    # Check Supabase connection
    print("🔍 Checking Supabase connection...")
    supabase_valid, supabase_error = validate_supabase_connection()
    if supabase_valid:
        passed_tests.append("✅ Supabase connection")
    else:
        errors.append(supabase_error)
        return False, errors, passed_tests  # Can't continue without Supabase
    
    # Check Supabase function
    print("🔍 Checking Supabase function...")
    function_valid, function_error = validate_supabase_function()
    if function_valid:
        passed_tests.append("✅ Supabase match_documents function")
    else:
        errors.append(function_error)
    
    # Check OpenAI connection
    print("🔍 Checking OpenAI connection...")
    openai_valid, openai_error = validate_openai_connection()
    if openai_valid:
        passed_tests.append("✅ OpenAI connection")
    else:
        errors.append(openai_error)
    
    return len(errors) == 0, errors, passed_tests


if __name__ == "__main__":
    """Run validation when script is executed directly."""
    print("🔍 Validating RAG Starter Kit setup...\n")
    
    all_valid, errors, passed_tests = validate_all()
    
    if all_valid:
        print("\n" + "=" * 60)
        print("✅ All validation checks passed!")
        print("=" * 60)
        for test in passed_tests:
            print(f"  {test}")
        
        # Check data directory (optional for validation)
        print("\n🔍 Checking data directory...")
        data_valid, data_error, file_count = validate_data_directory()
        if data_valid:
            print(f"✅ Data directory: Found {file_count} Markdown file(s) in ./data/")
        else:
            print(f"\n{data_error}")
    else:
        print("\n" + "=" * 60)
        print("❌ Setup validation failed")
        print("=" * 60)
        
        if passed_tests:
            print("\n✅ Passed checks:")
            for test in passed_tests:
                print(f"  {test}")
        
        print("\n❌ Failed checks:")
        for error in errors:
            print(f"\n{error}")
        
        print("\n💡 Fix the errors above and try again.")
        exit(1)
