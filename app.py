import streamlit as st
import pandas as pd
import boto3
import json


# =========================
# CONFIG & SETUP
# =========================
# Configure AWS Bedrock using Streamlit secrets
try:
    region = st.secrets.get("AWS_DEFAULT_REGION", "us-east-1")
    model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    
    # Initialize Bedrock client
    client = boto3.client(
        "bedrock-runtime",
        region_name=region,
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
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
# FUNCTION: Call Bedrock LLM
# =========================
def call_bedrock_llm(prompt):
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
# FUNCTION: Create Bedrock Prompt (FIXED TO MATCH SNOWFLAKE OUTPUT)
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
        # Create a string representation of the Python list for the prompt
        final_columns_list_str = ",\n        ".join([f"'{col}'" for col in target_columns_list])
       
        prompt = f"""
You are an expert Python data engineer specializing in pandas.
Generate a **complete, production-ready Python script** that reads 'source_data.csv' and performs all transformations described in <transformation_rules> automatically.

<goal>
- Intelligently interpret each transformation rule in natural language.
- Choose the most appropriate pandas methods automatically.
- Ensure type-safe operations — handle strings, numbers, dates, and nulls gracefully.
- **CRITICAL FINAL GOAL:** The final output must *only* contain the specific columns listed in <final_schema>. All other columns (junk, or old source columns) must be dropped.

<date_handling_strategy>
You MUST follow these specific strategies for any date-related transformations:

1.  **Import numpy:** The script MUST `import numpy as np` for null handling (`np.nan` or `pd.NA`).
2.  **Handle Invalid Values (CRITICAL):**
    * Before any parsing, you MUST replace known non-date strings (e.g., "Not Available", "Invalid", "", "NULL") with `pd.NA` (or `np.nan`).
    * `df['col'] = df['col'].replace(['Not Available', 'Invalid', '', 'NULL'], pd.NA, regex=False)`
    * After this, use `errors='coerce'` in `pd.to_datetime` to convert any remaining unparseable values to `NaT`.

3.  **Handle Mixed Formats (CRITICAL):**
    * Do NOT rely on automatic inference (`pd.to_datetime(col, errors='coerce')`) alone. This will fail on ambiguous formats.
    * You MUST use a **coalesce (fillna) strategy** by trying multiple explicit formats in order.
    * **Example Strategy:**
        ```python
        # First, clean known non-date strings
        df['temp'] = df['Source_Column'].replace(['Not Available', 'Invalid', '', 'NULL'], pd.NA, regex=False)
       
        # 1. Try standard inference (fastest)
        parsed = pd.to_datetime(df['temp'], errors='coerce')
       
        # 2. Fill NaNs by trying specific formats in order
        # Add ALL common formats, including the ones from the rules/images.
        parsed = parsed.fillna(pd.to_datetime(df['temp'], format='%Y-%m-%d', errors='coerce'))
        parsed = parsed.fillna(pd.to_datetime(df['temp'], format='%m/%d/%Y', errors='coerce'))
        parsed = parsed.fillna(pd.to_datetime(df['temp'], format='%d/%m/%Y', errors='coerce'))
        parsed = parsed.fillna(pd.to_datetime(df['temp'], format='%d-%m-%Y', errors='coerce'))
        parsed = parsed.fillna(pd.to_datetime(df['temp'], format='%Y.%m.%d', errors='coerce')) # For '2023.03.10'
        parsed = parsed.fillna(pd.to_datetime(df['temp'], format='%d.%m.%Y', errors='coerce')) # for '10.05.2021'
        parsed = parsed.fillna(pd.to_datetime(df['temp'], format='%d-%b-%y', errors='coerce')) # For '10-Mar-22'
        parsed = parsed.fillna(pd.to_datetime(df['temp'], format='%d %B %Y', errors='coerce')) # For '15 March 2024'
        parsed = parsed.fillna(pd.to_datetime(df['temp'], format='%B %d %Y', errors='coerce')) # For 'April 1 2024'
        parsed = parsed.fillna(pd.to_datetime(df['temp'], format='%Y-%m-%d %H:%M:%S', errors='coerce'))
        parsed = parsed.fillna(pd.to_datetime(df['temp'], format='%m/%d/%Y %H:%M', errors='coerce'))
       
        df['Target_Column'] = parsed
        ```

4.  **Handle Unix Timestamps:**
    * If a rule mentions "Unix timestamp" or data is numeric (e.g., 1710460800), you must convert using `pd.to_numeric(..., errors='coerce')` first, then `pd.to_datetime(..., unit='s', errors='coerce')`.
    * This should also be part of the `fillna` chain.
    * `parsed = parsed.fillna(pd.to_datetime(pd.to_numeric(df['temp'], errors='coerce'), unit='s', errors='coerce'))`

5.  **Remove Time:**
    * If a rule says "keep only date" (e.g., for 'Order_Date'), first parse to a full datetime object using the robust strategy above.
    * Then, extract the date component using `.dt.normalize()`. This keeps the `NaT` values intact.
    * `df['Target_Column'] = df['Parsed_Column'].dt.normalize()`

6.  **Final Formatting (YYYY-MM-DD) (CRITICAL FOR NULLS):**
    * If a rule requires a specific *string* format (like 'YYYY-MM-DD'), you apply this LAST.
    * Applying `.dt.strftime()` to a column with `NaT` converts `NaT` to the *string* "NaT", not a null.
    * **You MUST use `.where()` to preserve nulls.**
    * **Example Strategy:**
        ```python
        # df['Parsed_Date_Col'] is the result from step 3, 4, or 5
        # It contains datetime objects and NaT values
       
        # Convert to string, but ONLY for non-null values
        df['Target_Format'] = df['Parsed_Date_Col'].dt.strftime('%Y-%m-%d')
       
        # Restore NaNs where the original was null
        df['Target_Format'] = df['Target_Format'].where(df['Parsed_Date_Col'].notna(), pd.NA)
       

        df['Target_Format'] = df['Parsed_Date_Col'].dt.strftime('%m-%d-%Y')

        df['Target_Format'] = df['Target_Format'].where(df['Parsed_Date_Col'].notna(), pd.NA)


        ```
</date_handling_strategy>
</goal>

<requirements>
1.  Import `pandas` and `numpy`.
2.  Read 'source_data.csv' using `pd.read_csv(..., dtype=str)` to ensure all data is read as string/object initially. This prevents pandas from misinterpreting nulls or dates.
3.  Log progress for each rule using `print()`.
4.  Apply transformations based on the natural-language descriptions in <transformation_rules> and the <date_handling_strategy>.
5.  **Finalize DataFrame:** After all transformations, define the `final_schema` list as shown in <final_schema>.
6.  Select *only* the columns in that list using `df_final = df.reindex(columns=final_schema)`. This is critical to ensure the exact columns and order, and to drop all unwanted columns.
7.  Save `df_final` as 'target_output.csv' using `to_csv(index=False)`.
8.  Return only valid Python code inside a Python code block.
</requirements>

<source_schema>
Columns: {source_columns}
</source_schema>

<transformation_rules>
{rules_text}
</transformation_rules>

<final_schema>
# The final script MUST define this list and use it to reindex the DataFrame.
# This ensures only the desired target columns are in the output.
final_schema = [
        {final_columns_list_str}
]
</final_schema>

Return only Python code.
"""
    else:
        # #################################################################
        # ## THIS IS THE NEW, ROBUST PYSPARK PROMPT
        # #################################################################
       
        # Create a string representation for PySpark list
        final_columns_list_str = ", ".join([f'"{col}"' for col in target_columns_list])

        prompt = f"""
You are an expert PySpark data engineer.
Generate a **complete, production-ready PySpark script** that reads 'source_data.csv' and applies all transformations described in <transformation_rules> automatically using the Spark DataFrame API.

<goal>
- Intelligently interpret each transformation rule in natural language.
- Automatically choose the most appropriate `pyspark.sql.functions` (e.g., `F.when`, `F.concat`, `F.regexp_replace`, `F.to_timestamp`, `F.coalesce`, `F.year`, `F.date_format`, `F.to_date`).
- Ensure type-safe operations — handle strings, numbers, dates, and nulls gracefully.
- **CRITICAL FINAL GOAL:** The final output must *only* contain the specific columns listed in <final_schema>. All other columns must be dropped.

<general_transformation_strategy>
You MUST follow these specific strategies for common transformations:

1.  **Import Functions:** The script MUST `import pyspark.sql.functions as F`.
2.  **Null Handling (Enhanced):**
    * Before any operation, clean all columns using:
    ```python
    null_values = ["NA", "N/A", "NaN", "null", "NULL", "", "Not Available", "Invalid"]
    for colname in df.columns:
        df = df.withColumn(colname, F.when(F.col(colname).isin(null_values), None).otherwise(F.col(colname)))
    ```
3.  **Type Casting (Numeric):**
    * When converting strings to numbers, use `.cast("double")` or `.cast("integer")`:
    ```python
    df = df.withColumn("Target_Num", F.col("Source_Num").cast("double"))
    ```
4.  **String Manipulation:**
    * **Concatenation:** `F.concat_ws(" ", F.col("First"), F.col("Last"))`
    * **Splitting:** `F.split(F.col("Full_Name"), " ")[0]`
    * **Cleaning:** `F.trim()`, `F.lower()`, `F.upper()`
    * **Pattern Replacement:** `F.regexp_replace(F.col("col"), "pattern", "replacement")`
5.  **Conditional Logic (Case Statements):**
    * Use chained `F.when()`:
    ```python
    F.when(F.col("Status") == "A", F.lit("Active"))
     .when(F.col("Status") == "I", F.lit("Inactive"))
     .otherwise(F.lit("Unknown"))
    ```
6.  **Always wrap all `.withColumn(...)` chains in parentheses** for safety.

<date_handling_strategy>
1. **Robust Date Parsing, Cleaning, and Year Extraction**

```python
# 1️⃣ Clean invalid or null-like date values
cleaned_date_col = (
    F.when(F.col("Joining_Date").isin("Not Available", "Invalid", "NULL", "NA", "", "N/A", "NaN"), None)
     .otherwise(F.col("Joining_Date"))
)

# 2️⃣ Normalize all delimiters (. / _ \ or space) into '-'
cleaned_date_col = F.regexp_replace(cleaned_date_col, r"[./\\_ ]", "-")

# 3️⃣ Try all major date patterns using F.coalesce (auto-selects first successful parse)
parsed_date_col = F.coalesce(
    F.to_timestamp(cleaned_date_col, "yyyy-MM-dd HH:mm:ss"),
    F.to_timestamp(cleaned_date_col, "yyyy-MM-dd"),
    F.to_timestamp(cleaned_date_col, "MM-dd-yyyy"),
    F.to_timestamp(cleaned_date_col, "dd-MM-yyyy"),
    F.to_timestamp(cleaned_date_col, "yyyy.MM.dd"),
    F.to_timestamp(cleaned_date_col, "dd.MM.yyyy"),
    F.to_timestamp(cleaned_date_col, "yyyy/MM/dd"),
    F.to_timestamp(cleaned_date_col, "dd/MM/yyyy"),
    F.to_timestamp(cleaned_date_col, "dd-MMM-yy"),
    F.to_timestamp(cleaned_date_col, "dd-MMM-yyyy"),
    F.to_timestamp(cleaned_date_col, "dd MMM yyyy"),
    F.to_timestamp(cleaned_date_col, "dd MMMM yyyy"),
    F.to_timestamp(cleaned_date_col, "MMMM dd yyyy"),
    F.from_unixtime(F.col("Joining_Date").cast("long"))
)

# 4️⃣ Create normalized cleaned date column (yyyy-MM-dd)
df = df.withColumn("Cleaned_Joining_Date", F.date_format(parsed_date_col, "yyyy-MM-dd"))

# 5️⃣ Extract year from parsed date (e.g., Join_Year)
df = df.withColumn("Join_Year", F.year(parsed_date_col))

<requirements>
1. Initialize SparkSession.
2. Import `pyspark.sql.functions as F`.
3. Read the file using `spark.read.csv(..., header=True, inferSchema=False)`.
4. Normalize nulls across all columns using the enhanced null logic.
5. Apply each rule from <transformation_rules> using `.withColumn()` inside parentheses.
6. Define and apply `final_schema = [...]` for output selection.
7. Save final result as Parquet using:
    ```python
    df_final.write.mode("overwrite").parquet("target_output_parquet")
    ```
8. (Optional) Show output and export CSV for Google Colab:
    ```python
    df_check = spark.read.parquet("target_output_parquet")
    df_check.show(5)
    df_check.coalesce(1).write.mode("overwrite").option("header", True).csv("output_csv")
    import shutil, glob
    shutil.copy(glob.glob("output_csv/*.csv")[0], "/content/final_output.csv")
    ```

<source_schema>
Columns: {source_columns}
</source_schema>

<transformation_rules>
{rules_text}
</transformation_rules>

<final_schema>
final_schema = [
    {final_columns_list_str}
]
</final_schema>

Return only valid PySpark code.
"""

    return prompt

# -------------------------------------
# Streamlit App
# -------------------------------------
def main():
    st.info("""
    This app runs with **AWS Bedrock Claude**.
    Upload your **STTM Rules** file — Claude LLM will automatically infer your data schema 
    and generate a Python or PySpark script that performs all data-level transformations.
    """)
    
    # ---- Upload STTM File ----
    sttm_file = st.file_uploader("📤 Upload STTM Rules (CSV/XLSX)", type=["csv", "xlsx"])
    
    # ---- Select Script Type ----
    code_type = st.radio(
        "Select the script type you want to generate:",
        [" Python", " PySpark"],
        horizontal=True
    )
    code_type_flag = "python" if "Python" in code_type else "pyspark"
    
    # ---- Main Processing ----
    if sttm_file:
        try:
            rules_df = pd.read_excel(sttm_file) if sttm_file.name.endswith(".xlsx") else pd.read_csv(sttm_file)
            st.success("✅ STTM Rules file uploaded successfully!")

            with st.expander("📋 STTM Rules Preview", expanded=False):
                st.dataframe(rules_df)

            # Automatically infer source columns from STTM Rules
            source_columns = rules_df["Source_Column"].dropna().unique().tolist()
            source_df = pd.DataFrame(columns=source_columns)

            # Get target columns
            target_columns = rules_df["Target_Column"].dropna().unique().tolist()

            st.divider()
            st.subheader(f"🧠 Generate {code_type} Transformation Script")

            # Initialize session state
            if "generated_code" not in st.session_state:
                st.session_state.generated_code = None
                st.session_state.last_code_type = None

            if st.button(f"🚀 Generate {code_type} Script via Bedrock", use_container_width=True):
                with st.spinner(f"Calling AWS Bedrock Claude to generate {code_type}... please wait..."):
                    try:
                        prompt = create_bedrock_prompt(source_df, rules_df, target_columns, code_type=code_type_flag)
                        generated_code = call_bedrock_llm(prompt)

                        if "```" in generated_code:
                            generated_code = (
                                generated_code.split("```")[1]
                                .replace("python", "")
                                .replace("pyspark", "")
                                .strip()
                            )

                        # Save to session state to persist on reruns
                        st.session_state.generated_code = generated_code
                        st.session_state.last_code_type = code_type_flag
                        st.success(f"✅ {code_type} transformation script generated successfully!")

                    except Exception as e:
                        st.error(f"❌ Error while calling Bedrock: {e}")
                        st.error(f"Error type: {type(e).__name__}")

            # ✅ Keep showing generated code even after reruns or downloads
            if st.session_state.generated_code:
                st.subheader(f"📜 Generated {code_type} Script")
                st.code(st.session_state.generated_code, language="python")

                file_suffix = "pandas" if st.session_state.last_code_type == "python" else "pyspark"
                st.download_button(
                    label=f"💾 Download {code_type} Script (.py)",
                    data=st.session_state.generated_code.encode("utf-8"),
                    file_name=f"data_level_transformation_{file_suffix}.py",
                    mime="text/x-python",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"❌ Error processing uploaded file: {e}")

# Run the app
if __name__ == "__main__":
    main()
