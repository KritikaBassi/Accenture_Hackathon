# %%
import os
import warnings
import uuid
import json
import datetime
import re
import io
import csv
import gradio as gr
import pandas as pd
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print("Libraries installed and imported successfully.")

# %% [markdown]
# ## 1. Global Storage and Configuration (Same as v2)

# %%
ticket_database = {}
llm = None
retriever = None
rag_setup_successful = False
llm_initialized = False
rag_texts = []

SUPPORT_DEPARTMENTS = sorted(['Tier 1 Support', 'Tier 2 Support', 'Backend Engineering', 'IT Admin', 'Billing Department', 'Sales', 'Customer Success', 'Product Team', 'Knowledge Base Team', 'General Inquiry'])
ISSUE_CATEGORIES = sorted(['Software Installation', 'Network Connectivity', 'Device Compatibility', 'Account Access', 'Billing Inquiry', 'Payment Issue', 'Data Sync', 'Performance Issue', 'Feature Request', 'How-To Question', 'Bug Report', 'Security Concern', 'Other'])
SENTIMENTS = ['Positive', 'Neutral', 'Confused', 'Anxious', 'Frustrated', 'Annoyed', 'Urgent', 'Angry']
PRIORITIES = ['Low', 'Medium', 'High', 'Critical']
STATUS_OPTIONS = ['Open', 'In Progress', 'Pending Customer', 'Needs Follow-up', 'Resolved', 'Closed', 'Escalated']

HARDCODED_API_KEY = "AIzaSyBvLjAbdWz6v6_ii98B_j98IffX4TGrexM" # <--- REPLACE THIS
import os
os.environ["GOOGLE_API_KEY"] = HARDCODED_API_KEY
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
if HARDCODED_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
    print("\n" + "*"*70 + "\nWARNING: API Key placeholder detected. Replace 'YOUR_GEMINI_API_KEY_HERE'.\n" + "*"*70 + "\n")

def setup_rag_data():
    # (Keep the RAG data setup function exactly the same as in v2)
    global retriever, rag_setup_successful, rag_texts
    try:
        historical_data = [
             """KB_ID: KB001; Title: Resolving 'Unknown Error' during Software Installation; Keywords: install, update, unknown error, 75%, conflict; Content: This error often occurs due to interference from third-party antivirus software (e.g., Norton, McAfee, Avast). **Solution:** Temporarily disable the real-time protection of your antivirus, run the installer as administrator, and re-enable the antivirus after successful installation. If the issue persists, use the direct download link: [direct link placeholder]. Related Tickets: TKT001, TKT015; Team: Tier 1 Support""",
             """KB_ID: KB002; Title: Troubleshooting Dashboard Widget Slow Loading; Keywords: dashboard, widget, slow, performance, database; Content: Slow loading of specific dashboard widgets usually indicates a backend performance issue. **Troubleshooting:** 1. Clear browser cache/cookies. 2. Check network speed. 3. Verify if other parts of the application are slow. **Resolution:** If isolated to the widget, escalate to Backend Engineering with user account ID, widget name, and approximate time the issue started. They need to analyze database queries and server load. Related Tickets: TKT002; Team: Backend Engineering""",
             """Ticket ID: TKT004; Issue: Double Billing Charge; Resolution: Verified duplicate charge in Stripe. Issued a full refund for the second charge. Sent confirmation email with refund ID.; Team: Billing Department""",
             """KB_ID: KB003; Title: Connecting Google Calendar Integration; Keywords: google calendar, integration, connect, API key invalid; Content: To connect Google Calendar, go to Settings > Integrations > Calendar Sync. Click 'Connect'. If you receive an 'API Key Invalid' or authentication error: 1. Disconnect the integration if partially connected. 2. Go to your Google Account's 'Third-party apps with account access' settings. 3. Ensure our application has permission. If listed, try removing and re-adding access. 4. Attempt the connection again in our app. Related Tickets: TKT005; Team: Tier 2 Support"""
        ]
        data_dir = "historical_data_cache_v3" # Use same cache name
        os.makedirs(data_dir, exist_ok=True)
        for i, content in enumerate(historical_data):
            filename = f"hist_{i+1}.txt"
            entry_id = content.split(';')[0].split(':')[1].strip() if ':' in content.split(';')[0] else f"TKT_or_KB_{i+1}"
            with open(os.path.join(data_dir, filename), "w", encoding="utf-8") as f:
                f.write(f"Entry ID: {entry_id}\n{content}\n")
        print(f"Created/verified dummy historical data files in '{data_dir}'.")

        print("Setting up RAG pipeline structure...")
        loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
        documents = loader.load();
        if not documents: raise ValueError("No documents loaded for RAG.")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        texts = text_splitter.split_documents(documents);
        if not texts: raise ValueError("Document splitting failed.")
        print("RAG pipeline structure prepared.")
        rag_texts = texts
        return True
    except Exception as e:
        print(f"Error setting up RAG data: {e}")
        rag_setup_successful = False; retriever = None; rag_texts = []
        return False
rag_data_ready = setup_rag_data()


# %% [markdown]
# ## 2. Agent Definitions (Same as v2)

# %%
# --- All Prompt Definitions are identical to v2 ---
summarizer_prompt = PromptTemplate.from_template("""Provide a concise, neutral summary (3-4 sentences) of the core customer problem and the key interaction points.\n\nConversation History:\n{conversation_history}\n\nConcise Summary:""")
entity_prompt = PromptTemplate.from_template("""Analyze the conversation history and extract the following entities, if mentioned. Output *only* a valid JSON object with these keys (use null if information is not found):\n- "software_name": Name of the software/product being discussed.\n- "operating_system": Customer's OS (e.g., "Windows 11", "macOS Sonoma").\n- "error_message": Specific error text quoted or described by the user.\n- "version_number": Specific version of the software mentioned.\n- "hardware_involved": Any specific hardware mentioned (e.g., "thermostat model X").\n\nConversation History:\n{conversation_history}\n\nJSON Output:""")
entity_parser = JsonOutputParser()
category_prompt = PromptTemplate.from_template(f"""Assign the single MOST appropriate issue category from this list: {ISSUE_CATEGORIES}. Focus on the primary reason for contact. Output *only* the category name.\n\nConversation Summary:\n{{summary}}\n\nIssue Category:""")
sentiment_prompt = PromptTemplate.from_template(f"""Analyze the user's overall sentiment throughout the conversation. Choose the SINGLE most fitting sentiment from: {SENTIMENTS}. Output *only* the sentiment word.\n\nConversation History:\n{{conversation_history}}\n\nUser Sentiment:""")
priority_prompt = PromptTemplate.from_template(f"""Evaluate issue urgency/impact based on summary and sentiment. Assign ONE priority from: {PRIORITIES} (Critical, High, Medium, Low). Output *only* the priority word.\n\nGuidelines: Critical=System outage/security/payment failure; High=Major function broken/workflow blocked; Medium=Minor issue/inconvenience/how-to; Low=General query/feedback.\n\nConversation Summary:\n{{summary}}\nUser Sentiment:\n{{sentiment}}\n\nPriority Level:""")
action_extractor_prompt = PromptTemplate.from_template("""Extract specific, actionable tasks based on the conversation. List each distinct action using '- '. If none, state 'No specific actions identified'.\n\nConversation History:\n{conversation_history}\n\nExtracted Actions:""")
rag_prompt = PromptTemplate.from_template("""You are an AI Support Assistant providing resolution recommendations.\nBased on the Current Query and Context (similar past tickets/KB articles), suggest 1-3 concrete troubleshooting steps or solutions.\n\n**Instructions:**\n1.  Analyze the Context. If it contains a relevant solution (matching keywords/issue description), prioritize suggesting that.\n2.  Explicitly state if your suggestion is based on the provided Context (mentioning the Entry ID like KB_ID or Ticket ID if available in the context) or based on general knowledge if context is irrelevant/unavailable.\n3.  If Context is used and contains KB article info, suggest checking that specific article (e.g., "Based on context from KB001, try... You might also want to check Knowledge Base article KB001: [Title from context if available]").\n4.  If providing general steps, ensure they are logical for the issue described in the query.\n\nContext (Similar Past Tickets/KB Articles):\n{context}\n\nCurrent Customer Query/Conversation Summary:\n{question}\n\nRecommended Resolution Steps:""")
task_router_prompt = PromptTemplate.from_template(f"""Determine the single MOST appropriate primary team responsible for the *next step* or ownership based on the summary, category, and actions. Choose ONE team from: {SUPPORT_DEPARTMENTS}. Output *only* the department name.\n\nConversation Summary:\n{{summary}}\nIssue Category:\n{{category}}\nExtracted Actions:\n{{actions_list_str}}\n\nAssigned Department:""")
time_estimator_prompt = PromptTemplate.from_template("""Estimate the likely time for *initial response* or *resolution*. Provide a concise range (e.g., 'Under 1 hour', '1-4 business hours', '1-2 business days', 'Resolved during interaction').\n\nConversation Summary: {summary}\nCategory: {category}\nPriority: {priority}\nAssigned Department: {assigned_department}\n\nEstimated Resolution/Response Time:""")
status_suggester_prompt = PromptTemplate.from_template(f"""Analyze the *entire* conversation, especially the ending. Suggest the most appropriate final status for this ticket from the list: {STATUS_OPTIONS}. Consider if the issue was confirmed resolved, if the user needs to do something, or if further agent action is needed. Output *only* the suggested status word.\n\nConversation History:\n{{{{conversation_history}}}} \nSummary:\n{{{{summary}}}}\nExtracted Actions:\n{{{{actions_list_str}}}}\n\nSuggested Status:""") # Corrected braces here too
follow_up_prompt = PromptTemplate.from_template("""Based on the conversation outcome (summary, resolution, actions), suggest 1-2 concise, actionable follow-up tasks for the support team, if appropriate. Examples: "Check back with user in 2 days if status is still Pending Customer", "Update KB article KB001 with details about antivirus X conflict", "No follow-up needed". Output tasks as bullet points or 'No follow-up needed'.\n\nConversation Summary:\n{{summary}}\nRecommended Solution Provided:\n{{recommended_solution}}\nExtracted Actions:\n{{actions_list_str}}\nSuggested Status:\n{{suggested_status}}\n\nSuggested Follow-up Tasks:""")

# --- Helper Functions (Initialization, Agent Runners - Same as v2) ---
def initialize_gemini(api_key: str):
    # (Keep initialize_gemini function exactly the same as in v2)
    global llm, llm_initialized, embeddings, retriever, rag_setup_successful
    if llm_initialized and rag_setup_successful: return True
    if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE": return False
    try:
        print("Configuring & Initializing Gemini...")
        genai.configure(api_key=api_key)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, convert_system_message_to_human=True)
        llm.invoke("Test initialization.")
        llm_initialized = True
        print("Gemini LLM initialized.")
        if rag_data_ready and rag_texts:
            print("Initializing Embeddings & FAISS...")
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.from_documents(rag_texts, embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            rag_setup_successful = True
            print("RAG components initialized.")
        else:
             print("RAG data not ready, skipping RAG setup.")
             rag_setup_successful = False; retriever = None
        return True
    except Exception as e:
        print(f"Error initializing Gemini or RAG: {e}")
        llm_initialized = False; rag_setup_successful = False; llm = None; retriever = None
        return False

def run_agent(agent_name: str, prompt, llm_instance, parser=StrOutputParser(), **kwargs):
    # (Keep run_agent function exactly the same as in v2)
    if not llm_initialized or llm_instance is None: return f"Error: LLM N/A for {agent_name}"
    try:
        chain = prompt | llm_instance | parser
        result = chain.invoke(kwargs)
        if isinstance(result, (str, dict, list)):
             return result.strip() if isinstance(result, str) else result
        else:
             print(f"Warning: Agent '{agent_name}' returned unexpected type: {type(result)}. Converting to string.")
             return str(result).strip()
    except Exception as e:
        print(f"Error running agent '{agent_name}' (Prompt: {prompt.input_variables}, Parser: {type(parser).__name__}): {e}")
        error_detail = str(e)
        if "API key not valid" in error_detail: return f"Error: Invalid API Key for {agent_name}"
        elif "quota" in error_detail.lower(): return f"Error: API Quota Exceeded for {agent_name}"
        elif isinstance(parser, JsonOutputParser) and "parsing" in error_detail.lower(): return f"Error: Failed to parse JSON output for {agent_name}"
        return f"Error in {agent_name}"

def run_rag_agent(retriever_instance, llm_instance, rag_prompt_instance, query):
    # (Keep run_rag_agent function exactly the same as in v2)
    agent_name = "Resolution Recommender"
    if not llm_initialized or llm_instance is None: return f"Error: LLM N/A for {agent_name}"
    fallback_note = ""
    if not rag_setup_successful or retriever_instance is None:
        fallback_note = "\n\n(Note: RAG context unavailable. Providing general advice.)"
        print(f"Warning: RAG unavailable for {agent_name}.")
        general_prompt = PromptTemplate.from_template("Suggest general troubleshooting steps for: {question}")
        try:
            result = run_agent(f"{agent_name} (Fallback)", general_prompt, llm_instance, question=query)
            return result + fallback_note if "Error" not in result else result
        except Exception as e: return f"Error in {agent_name} (Fallback): {e}"
    try:
        rag_chain = ( {"context": retriever_instance | (lambda docs: "\n\n".join([f"Context Entry:\n{d.page_content}" for d in docs])), "question": RunnablePassthrough()} | rag_prompt | llm_instance | StrOutputParser() )
        result = rag_chain.invoke(query)
        return result.strip() if isinstance(result, str) else str(result)
    except Exception as e:
        print(f"Error running {agent_name} (RAG): {e}")
        fallback_note = "\n\n(Note: Error during RAG. Providing general advice.)"
        general_prompt = PromptTemplate.from_template("Suggest general troubleshooting steps for: {question}")
        try:
            result = run_agent(f"{agent_name} (Fallback after Error)", general_prompt, llm_instance, question=query)
            return result + fallback_note if "Error" not in result else result
        except Exception as fallback_e: return f"Error in {agent_name} (RAG and Fallback failed): {fallback_e}"


# --- Main Pipeline Orchestrator (MODIFIED to store raw outputs) ---
def process_customer_conversation(conversation_history: str):
    """
    Runs the full pipeline, stores processed AND RAW outputs, populates ticket.
    """
    init_success = initialize_gemini(HARDCODED_API_KEY)
    if not init_success:
        return {"error": True, "error_message": "LLM/RAG Initialization Failed."}

    print("\n--- Starting AI Support Pipeline ---")
    start_time = datetime.datetime.now()
    ticket = {'original_conversation': conversation_history} # Store raw input

    # --- Agent Execution & Raw Output Storage ---
    # Helper to run and store raw output
    def execute_and_store_raw(agent_key, agent_name, prompt, llm_inst, parser=StrOutputParser(), **kwargs):
        raw_output = run_agent(agent_name, prompt, llm_inst, parser=parser, **kwargs)
        ticket[f'raw_{agent_key}'] = raw_output # Store raw output
        # Return the potentially processed output (e.g., parsed JSON or stripped string)
        # For simplicity, we'll mostly use the raw output directly unless parsing fails
        if agent_key == 'entities' and not isinstance(raw_output, dict):
             # If entity parser failed, store error in processed field too
             ticket[agent_key] = {"error": f"Failed to parse entities: {raw_output}"}
        elif "Error" in raw_output and isinstance(raw_output, str):
             ticket[agent_key] = raw_output # Store error message in processed field
        else:
             ticket[agent_key] = raw_output # Store successful output
        print(f"{agent_key.capitalize()} Raw Output Stored.")
        return ticket[agent_key] # Return the processed/validated output for chaining

    print("1. Running Summarizer...")
    summary = execute_and_store_raw('summary', "Summarizer", summarizer_prompt, llm, conversation_history=conversation_history)
    if "Error" in summary: return {"error": True, "error_message": summary, **ticket}

    print("2. Running Entity Extractor...")
    entities = execute_and_store_raw('entities', "EntityExtractor", entity_prompt, llm, parser=entity_parser, conversation_history=conversation_history)
    # Error check already handled within execute_and_store_raw for entities
    if ticket['entities'].get("error"): return {"error": True, "error_message": ticket['entities']['error'], **ticket}

    print("3. Running Category Assigner...")
    category = execute_and_store_raw('issue_category', "CategoryAssigner", category_prompt, llm, summary=summary)
    if "Error" in category: return {"error": True, "error_message": category, **ticket}

    print("4. Running Sentiment Analyzer...")
    sentiment = execute_and_store_raw('sentiment', "SentimentAnalyzer", sentiment_prompt, llm, conversation_history=conversation_history)
    if "Error" in sentiment: return {"error": True, "error_message": sentiment, **ticket}

    print("5. Running Priority Assigner...")
    priority = execute_and_store_raw('priority', "PriorityAssigner", priority_prompt, llm, summary=summary, sentiment=sentiment)
    if "Error" in priority: return {"error": True, "error_message": priority, **ticket}

    print("6. Running Action Extractor...")
    actions = execute_and_store_raw('extracted_actions', "ActionExtractor", action_extractor_prompt, llm, conversation_history=conversation_history)
    if "Error" in actions: return {"error": True, "error_message": actions, **ticket}

    print("7. Running Resolution Recommender (RAG)...")
    # RAG runs slightly differently, store its raw output too
    recommended_solution = run_rag_agent(retriever, llm, rag_prompt, query=summary)
    ticket['raw_recommended_solution'] = recommended_solution # Store raw RAG output
    ticket['recommended_solution'] = recommended_solution # Use raw directly for processed field
    if "Error" in recommended_solution: print(f"Warning: RAG agent failed: {recommended_solution}")

    print("8. Running Task Router...")
    assigned_department = execute_and_store_raw('assigned_department', "TaskRouter", task_router_prompt, llm, summary=summary, category=category, actions_list_str=actions)
    if "Error" in assigned_department: return {"error": True, "error_message": assigned_department, **ticket}

    print("9. Running Time Estimator...")
    estimated_time = execute_and_store_raw('estimated_time', "TimeEstimator", time_estimator_prompt, llm, summary=summary, category=category, priority=priority, assigned_department=assigned_department)
    if "Error" in estimated_time: return {"error": True, "error_message": estimated_time, **ticket}

    print("10. Running Status Suggester...")
    suggested_status = execute_and_store_raw('suggested_status', "StatusSuggester", status_suggester_prompt, llm, conversation_history=conversation_history, summary=summary, actions_list_str=actions)
    if "Error" in suggested_status: return {"error": True, "error_message": suggested_status, **ticket}

    print("11. Running Follow-up Task Generator...")
    follow_up_tasks = execute_and_store_raw('suggested_follow_up', "FollowUpGenerator", follow_up_prompt, llm, summary=summary, recommended_solution=recommended_solution, actions_list_str=actions, suggested_status=suggested_status)
    if "Error" in follow_up_tasks: print(f"Warning: Follow-up generator failed: {follow_up_tasks}")

    # --- Final Ticket Assembly ---
    ticket['issue_id'] = f"ISS-{str(uuid.uuid4())[:8].upper()}"
    ticket['created_date'] = start_time.strftime("%Y-%m-%d %H:%M:%S")
    processed_status = ticket.get('suggested_status', 'Open')
    ticket['status'] = processed_status if processed_status in STATUS_OPTIONS else 'Open' # Validate suggested status
    ticket['resolution_date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") if ticket['status'] in ['Resolved', 'Closed'] else None
    ticket['final_solution'] = ticket.get('recommended_solution', None) if ticket['status'] in ['Resolved', 'Closed'] and "Error" not in ticket.get('recommended_solution', "") else None

    end_time = datetime.datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    ticket['ai_processing_time_sec'] = round(processing_time, 2)
    print(f"\n--- Pipeline Finished Successfully in {processing_time:.2f} seconds ---")
    ticket["error"] = False

    ticket_database[ticket['issue_id']] = ticket # Store the complete ticket with raw outputs
    print(f"Ticket {ticket['issue_id']} stored in memory.")
    return ticket


# --- Search, Dashboard & Export Functions (Dashboard/Export Unchanged from v2) ---
def format_tickets_for_dashboard(tickets_to_display: list) -> str:
    # (Keep format_tickets_for_dashboard function exactly the same as in v2)
    if not tickets_to_display: return "No tickets match filter criteria."
    cols = ['issue_id', 'created_date', 'issue_category', 'priority', 'status', 'assigned_department', 'sentiment']
    header = "| " + " | ".join(col.replace('_', ' ').title() for col in cols) + " |"
    sep = "|-" + "-|".join(['-' * len(col.replace('_', ' ').title()) for col in cols]) + "-|"
    rows = ["| " + " | ".join([str(ticket.get(col, 'N/A') or 'N/A') for col in cols]) + " |" for ticket in tickets_to_display]
    return f"{header}\n{sep}\n" + "\n".join(rows)

def get_dashboard_data(status_filter: str, department_filter: str, priority_filter: str):
    # (Keep get_dashboard_data function exactly the same as in v2)
    print(f"Filtering dashboard: Status='{status_filter}', Dept='{department_filter}', Prio='{priority_filter}'")
    filtered_tickets = []
    all_tickets = sorted(ticket_database.values(), key=lambda t: t['created_date'], reverse=True)
    for ticket in all_tickets:
        status_match = (status_filter == "All" or ticket.get('status') == status_filter)
        dept_match = (department_filter == "All" or ticket.get('assigned_department') == department_filter)
        prio_match = (priority_filter == "All" or ticket.get('priority') == priority_filter)
        if status_match and dept_match and prio_match: filtered_tickets.append(ticket)
    total_tickets = len(filtered_tickets)
    status_counts = pd.Series([t.get('status', 'Unknown') for t in filtered_tickets]).value_counts().to_dict()
    prio_counts = pd.Series([t.get('priority', 'Unknown') for t in filtered_tickets]).value_counts().to_dict()
    stats_md = f"**Displaying:** {total_tickets} tickets. "
    stats_md += f"**By Status:** {', '.join([f'{k}: {v}' for k, v in status_counts.items()])}. "
    stats_md += f"**By Priority:** {', '.join([f'{k}: {v}' for k, v in prio_counts.items()])}."
    table_md = format_tickets_for_dashboard(filtered_tickets)
    print(f"Found {total_tickets} matching tickets.")
    return stats_md, table_md

def export_tickets_to_csv():
    # (Keep export_tickets_to_csv function, but add raw keys to preferred_order)
    if not ticket_database: return None
    all_tickets = list(ticket_database.values())
    if not all_tickets: return None
    all_keys = set().union(*(d.keys() for d in all_tickets))
    preferred_order = [ # Add raw keys here
        'issue_id', 'created_date', 'status', 'priority', 'issue_category', 'sentiment',
        'assigned_department', 'estimated_time', 'summary', 'extracted_entities',
        'extracted_actions', 'recommended_solution', 'suggested_status',
        'suggested_follow_up', 'final_solution', 'resolution_date',
        'ai_processing_time_sec',
        # Raw Outputs
        'raw_summary', 'raw_entities', 'raw_issue_category', 'raw_sentiment',
        'raw_priority', 'raw_extracted_actions', 'raw_recommended_solution',
        'raw_assigned_department', 'raw_estimated_time', 'raw_suggested_status',
        'raw_suggested_follow_up'
        # Errors? Might be redundant if captured in raw outputs, but keep if needed
        # 'rag_error', 'follow_up_error'
    ]
    fieldnames = preferred_order + sorted(list(all_keys - set(preferred_order) - {'original_conversation'}))
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore', quoting=csv.QUOTE_ALL)
    writer.writeheader()
    for ticket in all_tickets:
         row_data = {}
         for key in fieldnames:
             value = ticket.get(key)
             if isinstance(value, (dict, list)): row_data[key] = json.dumps(value)
             else: row_data[key] = value
         writer.writerow(row_data)
    print(f"Prepared CSV data for {len(all_tickets)} tickets.")
    output.seek(0)
    return output.getvalue()

# --- NEW Function to display full ticket details ---
def display_ticket_details(issue_id: str):
    """Retrieves and formats ALL data for a specific ticket ID."""
    if not issue_id or issue_id == "N/A":
        return "No ticket ID selected."

    ticket_data = ticket_database.get(issue_id)
    if not ticket_data:
        return f"Ticket '{issue_id}' not found in current session."

    print(f"Formatting details for ticket: {issue_id}")
    # Define the order and formatting for display
    # Separate core fields, processed AI outputs, and raw AI outputs
    output_md = f"## Full Details for Ticket: {issue_id}\n\n"

    # --- Core Ticket Info ---
    output_md += "### Core Information\n"
    core_fields = ['created_date', 'status', 'priority', 'issue_category', 'sentiment', 'assigned_department', 'resolution_date', 'ai_processing_time_sec']
    for field in core_fields:
        output_md += f"- **{field.replace('_', ' ').title()}:** {ticket_data.get(field, 'N/A')}\n"
    output_md += "\n"

    # --- Processed AI Outputs ---
    output_md += "### Processed AI Analysis\n"
    processed_fields = ['summary', 'extracted_entities', 'extracted_actions', 'recommended_solution', 'estimated_time', 'suggested_status', 'suggested_follow_up', 'final_solution']
    for field in processed_fields:
        value = ticket_data.get(field, 'N/A')
        label = field.replace('_', ' ').title()
        # Format complex types like dicts/lists nicely
        if isinstance(value, dict):
            output_md += f"**{label}:**\n```json\n{json.dumps(value, indent=2)}\n```\n"
        elif isinstance(value, str) and '\n' in value: # Multi-line strings
            output_md += f"**{label}:**\n```\n{value}\n```\n"
        else:
            output_md += f"- **{label}:** {value}\n"
    output_md += "\n"

    # --- Raw AI Agent Outputs ---
    output_md += "### Raw AI Agent Outputs (for debugging/verification)\n"
    raw_keys = sorted([k for k in ticket_data.keys() if k.startswith('raw_')])
    for key in raw_keys:
         value = ticket_data.get(key)
         label = key.replace('raw_', '').replace('_', ' ').title()
         if isinstance(value, dict): # Raw entities might be dict
             output_md += f"**{label}:**\n```json\n{json.dumps(value, indent=2)}\n```\n"
         elif isinstance(value, str) and '\n' in value:
             output_md += f"**{label}:**\n```\n{value}\n```\n"
         else:
              output_md += f"- **{label}:** {value}\n"
    output_md += "\n"

    # --- Original Conversation (optional, maybe truncated) ---
    output_md += "### Original Conversation Snippet\n"
    convo = ticket_data.get('original_conversation', '')
    snippet = (convo[:500] + '...') if len(convo) > 500 else convo # Show snippet
    output_md += f"```\n{snippet}\n```\n"

    return output_md


# %% [markdown]
# ## 3. Gradio Interface Definition (Modified Search Tab)

# %%

def create_gradio_interface():
    """Creates the Gradio UI, including the modified Search tab."""

    # --- UI Helper Functions (process_and_display same as v2, update_dashboard same, handle_export same) ---
    def process_and_display(conversation_history):
        # (Keep process_and_display function exactly the same as in v2)
        # It now internally stores raw outputs via the modified process_customer_conversation
        print("\n=== Processing New Ticket Request ===")
        if not conversation_history:
            gr.Warning("Conversation history cannot be empty!")
            return ["Input Missing", "Provide conversation", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""] # Match number of outputs
        results = process_customer_conversation(conversation_history)
        if results.get("error", True):
            error_msg = results.get('error_message', 'Unknown processing error.')
            gr.Error(error_msg)
            return [results.get("issue_id", "N/A"), error_msg, "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""] # Match number of outputs
        else:
            print("=== Processing Complete, Updating UI ===")
            entities_str = json.dumps(results.get('entities', {}), indent=2) if isinstance(results.get('entities'), dict) else str(results.get('entities', '{}'))
            status_str = results.get('suggested_status', 'N/A')
            follow_up_str = results.get('suggested_follow_up', 'N/A')
            kb_suggestions = "N/A"
            if isinstance(results.get('recommended_solution'), str) and "Knowledge Base article" in results.get('recommended_solution', ""):
                 match = re.search(r"(KB\d+):?\s*([^.\n]+)?", results['recommended_solution'])
                 if match:
                     kb_suggestions = f"Check KB Article: {match.group(1)}" + (f" ({match.group(2).strip()})" if match.group(2) else "")
            ticket_summary_md = f"""### Ticket Created: {results['issue_id']}\n**Status:** {results['status']} | **Priority:** {results['priority']} | **Category:** {results['issue_category']}\n**Assigned Dept:** {results['assigned_department']} | **Est. Time:** {results['estimated_time']} | **Sentiment:** {results['sentiment']}\n\n**Summary:**\n{results['summary']}\n\n**AI Recommended Solution:**\n{results.get('recommended_solution', 'N/A')}"""
            # Match output list structure
            return [ entities_str, kb_suggestions, status_str, follow_up_str, results["issue_id"], ticket_summary_md,
                     results["summary"], results["issue_category"], results["sentiment"], results["priority"], results["extracted_actions"],
                     results.get("recommended_solution", "N/A"), results["assigned_department"], results["estimated_time"],
                     results.get('suggested_status', 'N/A'), results.get('suggested_follow_up', 'N/A'),
                     # Ensure raw entities is also stringified if needed for Textbox/Code output
                     json.dumps(results.get('entities', {}), indent=2) if isinstance(results.get('entities'), dict) else str(results.get('entities', '{}'))
                   ] # 17 outputs now


    def update_dashboard_display(status, department, priority):
        # (Keep update_dashboard_display function exactly the same as in v2)
        stats_md, table_md = get_dashboard_data(status, department, priority)
        return stats_md, table_md

    def handle_export():
        # (Keep handle_export function exactly the same as in v2)
        csv_data = export_tickets_to_csv()
        if csv_data:
            filename = f"ticket_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            return gr.File.update(value=csv_data, label=f"Download Export ({filename})", visible=True, interactive=False)
        else:
            gr.Info("No ticket data available to export.")
            return gr.File.update(visible=False)

    # --- NEW Search Handler ---
    def search_and_update_ui(issue_id_to_search: str):
        # Clean and standardize the input
        issue_id_raw = issue_id_to_search  # for debugging
        issue_id_to_search = issue_id_to_search.strip().upper()

        # Debug prints for console logs
        print(f"\n--- Search Initiated ---")
        print(f"Raw Input: '{issue_id_raw}'")
        print(f"Cleaned ID: '{issue_id_to_search}'")
        print(f"Current keys in ticket_database: {list(ticket_database.keys())}")

        if not issue_id_to_search:
            print("Search failed: No Issue ID entered.")
            return "Please enter an Issue ID.", None, gr.Button.update(visible=False), ""

        ticket_data = ticket_database.get(issue_id_to_search)
        if ticket_data:
            print(f"Search SUCCESS: Ticket '{issue_id_to_search}' found.")
            return (
                f"Ticket '{issue_id_to_search}' found. Click button below to view details.",
                issue_id_to_search,  # return plain value for state update
                gr.Button.update(visible=True),
                ""
            )
        else:
            print(f"Search FAILED: Ticket '{issue_id_to_search}' not found in database.")
            return (
                f"Ticket '{issue_id_to_search}' not found in the current session.",
                None,  # return plain None to clear the state
                gr.Button.update(visible=False),
                ""
            )


    # --- Build Gradio UI ---
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal", secondary_hue="cyan"), title="AI Support Assistant") as demo: # Changed theme slightly
        gr.Markdown("# Advanced AI Customer Support Assistant & Dashboard")
        gr.Markdown("")

        # Define State for Search Tab
        current_searched_id_state = gr.State(value=None)

        with gr.Tabs() as tabs:
            # --- TAB 1: Process New Conversation (Layout identical to v2) ---
            with gr.TabItem("Process New Conversation", id=0):
                 with gr.Row():
                    with gr.Column(scale=2):
                        conversation_input = gr.Textbox(label="Paste Full Conversation History", lines=20)
                        process_button = gr.Button("✨ Process & Analyze Conversation", variant="primary", size="lg")
                    with gr.Column(scale=3):
                        gr.Markdown("### 🤖 AI Agent Assist Panel")
                        with gr.Row():
                            assist_entities = gr.Code(label="Extracted Entities", language="json", interactive=False)
                            assist_kb = gr.Textbox(label="KB Suggestions", interactive=False)
                        with gr.Row():
                            assist_status = gr.Textbox(label="Suggested Status", interactive=False)
                            assist_followup = gr.Textbox(label="Suggested Follow-up", interactive=False, lines=2)
                        gr.Markdown("---")
                        gr.Markdown("### 🎟️ Generated Ticket Overview")
                        ticket_id_output = gr.Textbox(label="Ticket ID", interactive=False)
                        structured_ticket_output = gr.Markdown(label="Ticket Summary")
                 with gr.Accordion("Raw Agent Outputs (for debugging)", open=False):
                      summary_raw = gr.Textbox(label="Summary (Raw)", interactive=False, lines=2)
                      category_raw = gr.Textbox(label="Category (Raw)", interactive=False)
                      sentiment_raw = gr.Textbox(label="Sentiment (Raw)", interactive=False)
                      priority_raw = gr.Textbox(label="Priority (Raw)", interactive=False)
                      actions_raw = gr.Textbox(label="Actions (Raw)", interactive=False, lines=4)
                      resolution_raw = gr.Textbox(label="Resolution Reco (Raw)", interactive=False, lines=5)
                      routing_raw = gr.Textbox(label="Routing (Raw)", interactive=False)
                      time_raw = gr.Textbox(label="Time Est (Raw)", interactive=False)
                      status_raw = gr.Textbox(label="Status Suggestion (Raw)", interactive=False)
                      followup_raw = gr.Textbox(label="Follow-up Suggestion (Raw)", interactive=False, lines=2)
                      entities_raw_accordion = gr.Code(label="Entities (Raw JSON)", language="json", interactive=False) # Renamed to avoid clash

            # --- TAB 2: Ticket Dashboard (Layout identical to v2) ---
            with gr.TabItem("📊 Ticket Dashboard", id=1):
                gr.Markdown("### View, Filter, and Export Tickets")
                with gr.Row():
                    status_filter = gr.Dropdown(label="Status", choices=["All"] + STATUS_OPTIONS, value="All")
                    dept_filter = gr.Dropdown(label="Department", choices=["All"] + SUPPORT_DEPARTMENTS, value="All")
                    prio_filter = gr.Dropdown(label="Priority", choices=["All"] + PRIORITIES, value="All")
                dashboard_stats_md = gr.Markdown(label="Summary Statistics")
                dashboard_output_md = gr.Markdown(label="Filtered Ticket List")
                with gr.Row():
                     export_button = gr.Button("Export Current Tickets to CSV", variant="secondary")
                     export_file_output = gr.File(label="Download Export", visible=False, interactive=False)

            # --- TAB 3: Search Ticket (MODIFIED LAYOUT) ---
            with gr.TabItem("🔍 Search by ID", id=2):
                gr.Markdown("### Find a Specific Ticket by ID")
                with gr.Row():
                    search_id_input = gr.Textbox(label="Enter Issue ID", placeholder="e.g., ISS-1234ABCD", scale=3)
                    search_button = gr.Button("Search Ticket", scale=1)

                # Status message area
                search_status_output = gr.Markdown(label="Search Status")

                # "Show Details" button - initially hidden
                show_details_button = gr.Button("Show Full Ticket Details", visible=False, variant="primary")

                # Area to display the full details
                detailed_ticket_view_output = gr.Markdown(label="Ticket Details")


        # --- Connect UI elements to functions ---

        # Tab 1: Process Button Logic (Outputs match the number defined now)
        process_button.click(
            fn=process_and_display,
            inputs=[conversation_input],
            outputs=[ # 17 outputs: 4 assist + ID + summary + 11 accordion raw
                assist_entities, assist_kb, assist_status, assist_followup,
                ticket_id_output, structured_ticket_output,
                summary_raw, category_raw, sentiment_raw, priority_raw, actions_raw,
                resolution_raw, routing_raw, time_raw, status_raw, followup_raw, entities_raw_accordion
            ]
        )

        # Tab 2: Dashboard Filters & Export (Identical to v2)
        dashboard_filters = [status_filter, dept_filter, prio_filter]
        dashboard_outputs = [dashboard_stats_md, dashboard_output_md]
        for filter_component in dashboard_filters:
            filter_component.change(fn=update_dashboard_display, inputs=dashboard_filters, outputs=dashboard_outputs)
        demo.load(fn=update_dashboard_display, inputs=dashboard_filters, outputs=dashboard_outputs)
        export_button.click(fn=handle_export, inputs=[], outputs=[export_file_output])

        # Tab 3: Search Button Logic (NEW HANDLER)
        search_button.click(
            fn=search_and_update_ui, # Use the new handler
            inputs=[search_id_input],
            outputs=[
                search_status_output,       # Markdown for status message
                current_searched_id_state,  # Hidden State component
                show_details_button,        # Button component
                detailed_ticket_view_output # Markdown for details (cleared on search)
            ]
        )

        # Tab 3: Show Details Button Logic (NEW)
        show_details_button.click(
            fn=display_ticket_details,
            inputs=[current_searched_id_state], # Input is the ID stored in the state
            outputs=[detailed_ticket_view_output] # Output is the formatted details
        )

    return demo

# %%
if __name__ == '__main__':
    if not rag_data_ready: print("WARNING: RAG data not prepared.")
    if HARDCODED_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
         print("\n" + "="*70 + "\nFATAL ERROR: API Key is placeholder.\n" + "="*70 + "\n")
    else:
        print("Performing initial Gemini check...")
        init_success = initialize_gemini(HARDCODED_API_KEY)
        if not init_success:
             print("\n" + "="*70 + "\nFATAL ERROR: Initial Gemini check failed.\n" + "="*70 + "\n")
        else:
             print("Initial Gemini check successful.")
             gradio_app = create_gradio_interface()
             print("Launching Gradio Interface...")
             gradio_app.launch(share=True, inline=False)
