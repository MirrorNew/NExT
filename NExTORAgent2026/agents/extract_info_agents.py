import json
import re

from agents.base_agents import BaseAgent

class ExtractFactorsFromLongTextAgent(BaseAgent):
    def __init__(self, client, model_name="o3-mini", temperature=0.2):
        super().__init__(client, model_name, temperature)
        self.system_msg = (
            '''
            You are an expert mathematical modeler and an optimization professor at a top university. We will give you a description of an optimization problem.
            Regarding this combinatorial optimization problem, please complete the following tasks:
            Extract all decision variables and constraints from the following paragraph, ensuring that no element from the sentences is overlooked.

            1. **Sentence Scanning**: Start by providing the original sentence number and content, and then scan sentence by sentence. EITHER extract it into one or more constraints information OR mark it as "No constraints".
            2. **Variable List**: Give Variables from constraints sentence, and point: Name(symbol) / Meaning / type:<integer type OR continuous type>" / Range of Values. 
            3. **Mapping Table**: In a Markdown table, precisely correspond the "Constraint Name ↔ Mathematical Expression ↔ Sentence Number."
            4. **Optimization Goal**: Provide the optimization objective (target or performance metric to be optimized).
            5. **Problem Type**: Determine whether the model is a MILP (Mixed Integer Linear Programming) problem or an NLP (Nonlinear Programming) problem, and select one of the two values. 

            Note:
                1. List all variables, including those introduced for linearizing absolute differences, such as Δ⁺, Δ⁻ (if such variables exist, list them, otherwise leave them out), and generate the corresponding linearization constraints.  For each original sentence, scan and check if keywords like "change," "difference," "increment," "decrement," "change amount," etc., are mentioned, and generate the corresponding linearization constraints.
                2. "not need" does not necessarily mean that there are no variables or constraints. If an object appears in sentences such as "does not need to increase" or "will not increase", it may be necessary to consider the situation where the variable will decrease.
                3. Don’t ignore sentences starting with "In addition", "In addition to this", "By the way", etc., which may also contain information such as constraints or variables.
                4. If "all the sub-quotas" or "all types" are mentioned, then every category of situation must be considered.
            **Output** as follows:  
                  1.Sentence_Scanning
                    sentence 1:<sentence 1> -> <Constraint Scanning result description OR 'No constraints'>,
                    sentence 2:<sentence 2> -> <Constraint Scanning result description OR 'No constraints'>,
                    ...
                  2.Variables_List
                    Variable 1:...,
                    Variable 2:...,
                    ...
                  3.Constraint_Table
                    ["<Constraint 1 name>","<Mathematical expressions 1>","sentence numbers:<sentence numbers>"],
                    ["<Constraint 2 name>","<Mathematical expressions 2>","sentence numbers:<sentence numbers>"],
                    ...
                  4.Objective
                    <Objective sentence> and <Mathematical expressions>,
                    ...
                  5.Problem_Type
                    point <'MILP' OR 'NLP'>, and give description...
            '''
        )
        self.messages.append({"role": "system", "content": self.system_msg})

    async def generate(self, user_question, ):
        msgs = [
            {"role": "user", "content": (
                f'''
                    Here is the problem description:
                    ________________________________________
                    {user_question}
                    Output the lists(Sentence_Scanning, Variables_List, Constraint_Table,Objective, Problem_Type) mentioned above.'''
            )}
        ]
        self.messages.extend(msgs)
        return await self._query()

    async def get_and_change_format_output(self, origin_reply):
        self.messages.append({"role": "assistant", "content": (
            f"{origin_reply}"
        )})
        msgs = [
            {"role": "user", "content": (
                '''
                Now, please convert all your analysis results into the following JSON object format (no additional text):  
                For origin sentence, just write the first few words and "...".
                ```json
                {
                  "Sentence_Scanning": [
                    ["1","<origin sentence 1...>","<Constraint Scanning result description OR 'No constraints'>"],
                    ["2","<origin sentence 2...>","<Constraint Scanning result description OR 'No constraints'>"],
                    ...
                  ],
                  "Variables_List": [
                    {
                      "symbol":     "<chosen mathematical symbol>",
                      "Meaning":    "<parameter definition>",
                      "Type":       "<BINARY / integer / continuous type>",
                      "Range ":     "<Range of Values>"
                    },
                    ...
                  ],
                  "Constraint_Table":[
                    ["<Constraint 1 name>","<Mathematical expressions 1>","sentence numbers:<sentence numbers>"],
                    ["<Constraint 2 name>","<Mathematical expressions 2>","sentence numbers:<sentence numbers>"],
                    ...
                  ],
                  "Objective": {
                      "Objective_sentence":        "<Objective sentence>",
                      "Mathematical_expressions":  "<Mathematical expressions>"
                  },
                  "Problem_Type": "<'MILP' OR 'NLP'>"
                }```
            '''
            )}
        ]
        self.messages.extend(msgs)
        return await self._query()

    async def integrate_with_file(self, entry):
        """
        Load questions and answers from an existing JSON file, generate background, constraints, objective for each entry,
        merge them under respective keys, and write to a new JSON file.
        """
        question = entry.get('question', '')
        # Generate background, constraints, and objective via LLM
        try:
            origin_reply = await self.generate(question)
            origin_reply_format = await self.get_and_change_format_output(origin_reply)
            origin_reply_format_json = re.sub(r'```json|```', '', origin_reply_format).strip()
            targets_data = json.loads(origin_reply_format_json)
            # Merge the output with the existing entry
            entry['Sentence_Scanning'] = targets_data.get('Sentence_Scanning', [])
            entry['Variables_List'] = targets_data.get('Variables_List', [])
            entry['Constraint_Table'] = targets_data.get('Constraint_Table', [])
            entry['Objective'] = targets_data.get('Objective', {})
            entry['Problem_Type'] = targets_data.get('Problem_Type', "")
            print(f"Success ExtractFactorsFromLongTextAgent For Case {entry['index']}!")

        except Exception as e:
            # In case of error, log and continue
            print(f"Error generating Factors for Entry {entry['index']}: {e}")
        return entry


class ExtractParameterFromLongTextAgent(BaseAgent):
    def __init__(self, client, model_name="o3-mini", temperature=0.2):
        super().__init__(client, model_name, temperature)
        self.system_msg = (
            '''
            You are an expert in mathematical modeling and a professor of optimization at a top university. We will describe an optimization problem for you. Regarding this combinatorial optimization problem, please complete the following tasks:

            1. **Sentence Scanning**: Start by providing the original sentence number and content, and then scan sentence by sentence. EITHER extract it into one or more constraints information OR  just mark it as "No Values".
            2. **Extract Parameters**: Extract all parameters from the following paragraphs and tables, making sure that no elements in any sentence are omitted. Specifically, you need to give a **parameter list**, provide the names of all parameters, and must indicate the parameter type (integer/float/list/tuple) and give specific values.

            Note 1. The "Value" of the list/tuple type are defined by using the python format, and should not be string. Example, a list can use ["S", "V"] or a tuple type can use {"A": 450, "B": 400, "C": 300} and so on.
            Note 2. If the problem description contains **table** data (usually in markdown format), please strictly convert the table data into the form of a list or tuple in the python language. You must strictly refer to the data I provide and do not make up your own data. In the end, you must also extract the table data and name it Table_1_XXX, Table_2_XXX, and so on.
            Note 3. The step you are processing now is only used to find parameters with specific values, and you do not need to consider decision variables or other constraints!

            **Output** as follows:  
            1.Sentence Scanning Result
                sentence 1:<sentence 1> -> <Constraint Scanning result description OR 'No Values'>,
                sentence 2:<sentence 2> -> <Constraint Scanning result description OR 'No Values'>,
                ...
            2.Table Scanning Result
                table 1:<table_1_name> -> <Parameter Values(list/tuple)>,
                table 2:<table_2_name> -> <Parameter Values(list/tuple)>,
                ...
            '''
        )
        self.messages.append({"role": "system", "content": self.system_msg})

    async def generate(self, user_question, ):
        msgs = [
            {"role": "user", "content": (
                f'''
                    Here is the problem description:
                    ________________________________________
                    {user_question}
                    Output the result mentioned above.'''
            )}
        ]
        self.messages.extend(msgs)
        return await self._query()

    async def get_and_change_format_output(self, origin_reply):
        self.messages.append({"role": "assistant", "content": (
            f"{origin_reply}"
        )})
        msgs = [
            {"role": "user", "content": (
                '''
                Now, please convert all your analysis results(Sentence Parameters and Table Parameters) into **Parameters List**.

                Please adhere **strictly** to the following rules when generating the JSON field **"Value"**:
                1. Output must be **valid JSON**:  
                • All keys and string values in double quotes.  
                • No Python tuple syntax `(a, b)`.  
                • No objects with numeric or tuple keys.
                2.Value must follows these rules:  
                • If the original key is a **string**, keep the object structure.
                • If the original key is an **integer** (`0,1,2,…`), output a **one‐dimensional** array. Element at index i corresponds to the value for key i.  
                • If the original key is an **integer pair** `[i,j]`, output a **two‐dimensional square matrix**:  

                 Now, use the following JSON object format (no additional text):  
                ```json
                {
                    "Parameters_List": [
                        {
                            "Name": "<Name of parameter1>",
                            "Type": "<integer/float/list/tuple>",
                            "Value": <Parameter Values, not string>,
                        },
                        ...
                    ]
                }
                '''
            )}
        ]
        self.messages.extend(msgs)
        return await self._query()

    async def integrate_with_file(self, entry):
        """
        Load questions and answers from an existing JSON file, generate background, constraints, objective for each entry,
        merge them under respective keys, and write to a new JSON file.
        """
        question = entry.get('question', '')
        # Generate background, constraints, and objective via LLM
        try:
            origin_reply = await self.generate(question)
            origin_reply_format = await self.get_and_change_format_output(origin_reply)
            origin_reply_format_json = re.sub(r'```json|```', '', origin_reply_format).strip()

            # origin_reply_format_json_fixed = quote_value_section(origin_reply_format_json)
            # origin_reply_format_json_fixed_fixed = fix_value_section(origin_reply_format_json_fixed)
            # targets_data = json.loads(origin_reply_format_json_fixed_fixed)
            targets_data = json.loads(origin_reply_format_json)

            # Merge the output with the existing entry
            entry['Parameters_List'] = targets_data.get('Parameters_List', [])
            print(f"Success ExtractParameter For Case {entry['index']}!")
        except Exception as e:
            # In case of error, log and continue
            print(f"Error generating Parameters for Entry {entry['index']}: {e}")
        return entry





class DirectExtractALL(BaseAgent):
    def __init__(self, client, model_name="gemini-3-flash-preview-high", temperature=0.2):
        super().__init__(client, model_name, temperature)
        self.system_msg = (
            '''
            You are an expert mathematical modeler and an optimization professor.
            Read the following Operations Research (OR) problem description in its entirety.

            Extract all elements directly and output a structured JSON containing:
            1. "Parameters_List": List of given constants/parameters (Name, Meaning, Type, Value).
            2. "Variables_List": List of decision variables (Symbol, Meaning, Type, Range).
            3. "Constraint_Table": List of constraints (Constraint Name, Mathematical expressions).
            4. "Objective": The optimization goal (Objective_sentence, Mathematical_expressions).
            5. "Problem_Type": 'MILP' or 'NLP'.

            Do not extract sentence by sentence. Extract the global information directly.
            Output strictly in JSON format without any markdown wrappers or additional text.
            '''
        )
        self.messages.append({"role": "system", "content": self.system_msg})

    async def generate(self, user_question, ):
        msgs = [
            {"role": "user", "content": (
                f'''
                    Here is the problem description:
                    ________________________________________
                    {user_question}
                    Output the result mentioned above.'''
            )}
        ]
        self.messages.extend(msgs)
        result_text = await self._query()
        try:
            return json.loads(result_text)
        except Exception as e:
            print(f"Error json.loads: {e}")
            return result_text