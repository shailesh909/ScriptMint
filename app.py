import streamlit as st
import pandas as pd
import boto3
import json
import os
import io
# from dotenv import load_dotenv

# =========================
# CONFIG & SETUP
# =========================
try:
    region = st.secrets.get("AWS", {}).get("DEFAULT_REGION", "us-east-1")
    model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    
    # Initialize Bedrock client using Streamlit secrets
    client = boto3.client(
        "bedrock-runtime",
        region_name=region,
        aws_access_key_id=st.secrets["AWS"]["ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS"]["SECRET_ACCESS_KEY"]
    )
except Exception as e:
    st.error(f"AWS Bedrock configuration error: {e}")
    client = None

st.set_page_config(
    page_title="ScriptMint AI — Bedrock Data Transformer", 
    layout="wide",
    page_icon="🔧"
)
st.title("🔧 ScriptMint AI - Code Generator")

# =========================
# BEDROCK CALL FUNCTION
# =========================
def call_bedrock(prompt):
    """Call AWS Bedrock with the given prompt"""
    if client is None:
        return "Error: AWS Bedrock client not configured properly. Please check your AWS credentials."
    
    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4000,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}]
            })
        )
        output = json.loads(response["body"].read())
        return output["content"][0]["text"]
    except Exception as e:
        return f"Error calling Bedrock: {str(e)}"

# =========================
# PROMPT CREATION FUNCTION
# =========================
def create_bedrock_prompt(source_df, rules_df, target_columns_list, code_type="python"):
   
    source_columns = ", ".join(source_df.columns.tolist())

    rules_text = ""
    for _, rule in rules_df.iterrows():
        src = rule.get("Source_Column", "N/A")
        tgt = rule.get("Target_Column", "N/A")
        desc = rule.get("Transformation_Rule", "N/A")
        rules_text += f"- From column '{src}' → '{tgt}' | Transformation: {desc}\n"

    if code_type == "python":
        final_columns_list_str = ",\n        ".join([f"'{col}'" for col in target_columns_list])
       
        prompt = f"""You are an expert Python data engineer specializing in pandas.
Generate a complete, production-ready Python script that reads 'source_data.csv' and performs all transformations described in the rules automatically.

GOAL:
- Intelligently interpret each transformation rule in natural language
- Choose the most appropriate pandas methods automatically
- Ensure type-safe operations — handle strings, numbers, dates, and nulls gracefully
- CRITICAL FINAL GOAL: The final output must ONLY contain the specific columns listed in final_schema

DATE HANDLING STRATEGY:
1. Import numpy: The script MUST import numpy as np for null handling
2. Handle Invalid Values: Replace known non-date strings with pd.NA before parsing
3. Handle Mixed Formats: Use coalesce strategy trying multiple explicit formats in order
4. Handle Unix Timestamps: Convert using pd.to_numeric() first, then pd.to_datetime()
5. Remove Time: Use .dt.normalize() for date-only fields
6. Final Formatting: Use .where() to preserve nulls when formatting dates as strings

REQUIREMENTS:
1. Import pandas and numpy
2. Read source_data.csv using pd.read_csv(..., dtype=str)
3. Log progress for each rule using print()
4. Apply transformations based on natural-language descriptions
5. Define final_schema list and use df.reindex(columns=final_schema)
6. Save as target_output.csv using to_csv(index=False)
7. Return only valid Python code inside a Python code block

SOURCE SCHEMA: {source_columns}

TRANSFORMATION RULES:
{rules_text}

FINAL SCHEMA:
final_schema = [
        {final_columns_list_str}
]

Return only Python code without any explanations or markdown formatting."""

    else:
        # PySpark version
        final_columns_list_str = ", ".join([f'"{col}"' for col in target_columns_list])

        prompt = f"""You are an expert PySpark data engineer.
Generate a complete, production-ready PySpark script that reads 'source_data.csv' and applies all transformations using Spark DataFrame API.

GOAL:
- Intelligently interpret each transformation rule in natural language
- Automatically choose appropriate pyspark.sql.functions
- Ensure type-safe operations with proper null handling
- CRITICAL: Final output must ONLY contain columns listed in final_schema

GENERAL TRANSFORMATION STRATEGY:
1. Import pyspark.sql.functions as F
2. Enhanced null handling: clean all columns first
3. Type casting: use .cast('double') or .cast('integer') for numeric conversions
4. String manipulation: use F.concat_ws, F.split, F.trim, F.regexp_replace
5. Conditional logic: use chained F.when() statements
6. Always wrap .withColumn() chains in parentheses

DATE HANDLING STRATEGY:
1. Clean invalid date values first
2. Normalize delimiters to '-'
3. Try multiple date patterns using F.coalesce
4. Extract components like year using F.year()

REQUIREMENTS:
1. Initialize SparkSession
2. Import pyspark.sql.functions as F
3. Read file using spark.read.csv(..., header=True, inferSchema=False)
4. Normalize nulls across all columns
5. Apply each rule using .withColumn()
6. Define and apply final_schema for output selection
7. Save final result as Parquet

SOURCE SCHEMA: {source_columns}

TRANSFORMATION RULES:
{rules_text}

FINAL SCHEMA:
final_schema = [
    {final_columns_list_str}
]

Return only valid PySpark code without any explanations or markdown formatting."""

    return prompt

# =========================
# MAIN APPLICATION
# =========================
def main():
    st.sidebar.header("About")
    st.sidebar.info(
        "This app uses AWS Bedrock with Claude 3.5 Sonnet to generate "
        "data transformation scripts from STTM rules files."
    )
    
    st.sidebar.header("Requirements")
    st.sidebar.markdown("""
    - AWS credentials configured
    - STTM Rules file (CSV/Excel)
    - Bedrock access to Claude 3.5 Sonnet
    """)
    
    st.info("""
    **Local AWS Bedrock Version**  
    Upload your **STTM Rules** file — Bedrock LLM will automatically infer your data schema 
    and generate a Python or PySpark script that performs all data-level transformations.
    """)
    
    # Check AWS configuration
    if client is None:
        st.error("⚠️ AWS Bedrock not configured. Please check your AWS credentials in the .env file.")
        st.markdown("""
        **Setup Instructions:**
        1. Create a `.env` file with:
           ```
           AWS_ACCESS_KEY_ID=your_access_key
           AWS_SECRET_ACCESS_KEY=your_secret_key
           AWS_DEFAULT_REGION=us-east-1
           ```
        2. Or run `aws configure` in your terminal
        3. Ensure you have Bedrock access for Claude 3.5 Sonnet
        """)
        return
    
    # ---- Upload STTM File ----
    sttm_file = st.file_uploader(
        "📤 Upload STTM Rules (CSV/XLSX)", 
        type=["csv", "xlsx"],
        help="Upload your Source-to-Target Mapping rules file"
    )
    
    # ---- Select Script Type ----
    code_type = st.radio(
        "Select the script type you want to generate:",
        ["Python", "PySpark"],
        horizontal=True
    )
    code_type_flag = "python" if "Python" in code_type else "pyspark"
    
    # ---- Main Processing ----
    if sttm_file:
        try:
            # Read the uploaded file
            if sttm_file.name.endswith(".xlsx"):
                rules_df = pd.read_excel(sttm_file)
            else:
                rules_df = pd.read_csv(sttm_file)
                
            st.success(f"✅ STTM Rules file uploaded successfully! ({len(rules_df)} rules found)")

            # Preview rules
            with st.expander("📋 STTM Rules Preview", expanded=False):
                st.dataframe(rules_df, use_container_width=True)

            # Automatically infer source columns from STTM Rules
            source_columns = rules_df["Source_Column"].dropna().unique().tolist()
            source_df = pd.DataFrame(columns=source_columns)

            # Get target columns
            target_columns = rules_df["Target_Column"].dropna().unique().tolist()
            
            st.write(f"**Detected:** {len(source_columns)} source columns → {len(target_columns)} target columns")

            st.divider()
            st.subheader(f"🧠 Generate {code_type} Transformation Script")

            # Initialize session state
            if "generated_code" not in st.session_state:
                st.session_state.generated_code = None
                st.session_state.last_code_type = None

            generate_button = st.button(
                f"🚀 Generate {code_type} Script via Bedrock", 
                use_container_width=True,
                type="primary"
            )
            
            if generate_button:
                with st.spinner(f"Calling AWS Bedrock to generate {code_type} script... This may take 20-30 seconds..."):
                    try:
                        prompt = create_bedrock_prompt(source_df, rules_df, target_columns, code_type=code_type_flag)
                        
                        # Show prompt preview
                        with st.expander("📝 Prompt Preview", expanded=False):
                            st.text_area("Bedrock Prompt", prompt, height=200, label_visibility="collapsed")
                        
                        generated_code = call_bedrock(prompt)

                        # Extract code from markdown code blocks if present
                        if "```" in generated_code:
                            parts = generated_code.split("```")
                            if len(parts) >= 2:
                                generated_code = parts[1].replace("python", "").replace("pyspark", "").strip()

                        # Save to session state to persist on reruns
                        st.session_state.generated_code = generated_code
                        st.session_state.last_code_type = code_type_flag
                        st.success(f"✅ {code_type} transformation script generated successfully!")

                    except Exception as e:
                        st.error(f"❌ Error while calling Bedrock: {e}")
                        st.error(f"Error type: {type(e).__name__}")

            # Show generated code if available
            if st.session_state.generated_code and st.session_state.last_code_type == code_type_flag:
                st.subheader(f"📜 Generated {code_type} Script")
                
                # Code editor with copy button
                st.code(st.session_state.generated_code, language="python")
                
                # Download button
                file_suffix = "pandas" if code_type_flag == "python" else "pyspark"
                st.download_button(
                    label=f"💾 Download {code_type} Script (.py)",
                    data=st.session_state.generated_code.encode("utf-8"),
                    file_name=f"data_transformation_{file_suffix}.py",
                    mime="text/x-python",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"❌ Error processing uploaded file: {e}")
            st.info("Please ensure your file has the required columns: 'Source_Column', 'Target_Column', and 'Transformation_Rule'")

# Run the app
if __name__ == "__main__":
    main()
