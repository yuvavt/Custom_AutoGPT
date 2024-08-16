from __future__ import annotations

import json
import platform
import re
from logging import Logger
import os
from dotenv import load_dotenv
import distro
from forge.config.ai_directives import AIDirectives
from forge.config.ai_profile import AIProfile
from forge.json.parsing import extract_dict_from_json
from forge.llm.prompting import ChatPrompt, LanguageModelClassification, PromptStrategy
from forge.llm.prompting.utils import format_numbered_list
from forge.llm.providers.schema import (
    AssistantChatMessage,
    ChatMessage,
    CompletionModelFunction,
)
from forge.models.action import ActionProposal
from forge.models.config import SystemConfiguration, UserConfigurable
from forge.models.json_schema import JSONSchema
from forge.models.utils import ModelWithSummary
from forge.utils.exceptions import InvalidAgentResponseError
from pydantic import Field
from typing import List, Tuple, Dict, Any

_RESPONSE_INTERFACE_NAME = "AssistantResponse"

class AssistantThoughts(ModelWithSummary):
    observations: str = Field(
        description="Relevant observations from your last action (if any)"
    )
    text: str = Field(description="Thoughts")
    reasoning: str = Field(description="Reasoning behind the thoughts")
    self_criticism: str = Field(description="Constructive self-criticism")
    plan: list[str] = Field(description="Short list that conveys the long-term plan")
    speak: str = Field(description="Summary of thoughts, to say to user")

    def summary(self) -> str:
        return self.text


class OneShotAgentActionProposal(ActionProposal):
    thoughts: AssistantThoughts  # type: ignore


class OneShotAgentPromptConfiguration(SystemConfiguration):
    DEFAULT_BODY_TEMPLATE: str = (
        "## Constraints\n"
        "You operate within the following constraints:\n"
        "{constraints}\n"
        "\n"
        "## Resources\n"
        "You can leverage access to the following resources:\n"
        "{resources}\n"
        "\n"
        "## Commands\n"
        "These are the ONLY commands you can use."
        " Any action you perform must be possible through one of these commands:\n"
        "{commands}\n"
        "\n"
        "## Best practices\n"
        "{best_practices}"
        "\n"
        "{materials_science_info}"
    )

    DEFAULT_CHOOSE_ACTION_INSTRUCTION: str = (
        "Determine exactly one command to use next based on the given goals "
        "and the progress you have made so far, "
        "and respond using the JSON schema specified previously:"
    )

    DEFAULT_MATERIALS_SCIENCE_INFO: str = (
        "## Materials Science Capabilities\n"
        "You have access to the Materials Project database for materials science queries.\n"
        "For materials science related questions, prioritize using the Materials Project API.\n"
        "Use the 'query_materials_project' command for such queries."
    )

    body_template: str = UserConfigurable(default=DEFAULT_BODY_TEMPLATE)
    choose_action_instruction: str = UserConfigurable(
        default=DEFAULT_CHOOSE_ACTION_INSTRUCTION
    )
    use_functions_api: bool = UserConfigurable(default=False)
    enable_materials_science: bool = UserConfigurable(default=False)
    materials_science_info: str = UserConfigurable(default=DEFAULT_MATERIALS_SCIENCE_INFO)

    #########
    # State #
    #########
    # progress_summaries: dict[tuple[int, int], str] = Field(
    #     default_factory=lambda: {(0, 0): ""}
    # )

    def get_formatted_body(self, **kwargs) -> str:
        materials_science_info = self.materials_science_info if self.enable_materials_science else ""
        return self.body_template.format(
            materials_science_info=materials_science_info,
            **kwargs
        )


class OneShotAgentPromptStrategy(PromptStrategy):
    default_configuration: OneShotAgentPromptConfiguration = (
        OneShotAgentPromptConfiguration()
    )
    #init method
    def __init__(
        self,
        configuration: OneShotAgentPromptConfiguration,
        logger: Logger,
    ):
        self.config = configuration
        self.response_schema = JSONSchema.from_dict(
            OneShotAgentActionProposal.model_json_schema()
        )
        self.logger = logger
        self.material_keywords = [
            "material", "element", "compound", "alloy", "crystal",
            "metal", "ceramic", "polymer", "semiconductor", "properties",
            "structure", "synthesis", "characterization", "bandgap",
            "conductivity", "strength", "ductility", "hardness", "phase",
            "lattice", "doping", "thin film", "nanostructure"
        ]

        # Load environment variables
        load_dotenv()
        
        # Get Materials Project API key from environment variable
        self.materials_project_api_key = os.getenv('MATERIALS_PROJECT_API_KEY')
        
        if not self.materials_project_api_key:
            self.logger.warning("Materials Project API key not found in .env file.")


    #Method to check if the query is material related or not
    def is_materials_science_query(self, query: str) -> bool:
        """
        Determine if the given query is related to materials science.
        
        Args:
        query (str): The input query to check.
        
        Returns:
        bool: True if the query is related to materials science, False otherwise.
        """
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.material_keywords)
    

    
    def generate_materials_science_prompt(self, task: str, context: Optional[str] = None) -> str:
        prompt = f"""
        TASK: {task}

        THOUGHTS: This is a materials science-related query. I should use the Materials Project database to find accurate and up-to-date information.

        REASONING: The Materials Project database is a comprehensive resource for materials science data, which is more suitable for this type of query than a general web search.

        PLAN:
        1. Identify the specific material or property mentioned in the query.
        2. Determine the appropriate Materials Project API endpoint and parameters.
        3. Make a request to the Materials Project database.
        4. Analyze the response and extract relevant information.
        5. Provide a concise answer based on the retrieved data.

        CRITICISM: I must ensure that I correctly interpret the query and use the appropriate API endpoint. If the data is not available in the Materials Project database, I should have a fallback plan to use other reliable sources.

        NEXT ACTION: query_materials_project

        RESPONSE: Based on the materials science query, I will use the Materials Project API to find the requested information. The next step is to query the database with the specific details extracted from the task.
        """
        return prompt

    @property
    def llm_classification(self) -> LanguageModelClassification:
        return LanguageModelClassification.FAST_MODEL  # FIXME: dynamic switching

    def build_prompt(
        self,
        *,
        messages: List[ChatMessage],
        task: str,
        ai_profile: AIProfile,
        ai_directives: AIDirectives,
        commands: List[CompletionModelFunction],
        include_os_info: bool,
        **extras,
    ) -> ChatPrompt:
        """Constructs and returns a prompt with the following structure:
        1. System prompt
        2. Task description
        3. Previous messages
        4. Final instruction
        """
        # Check if the task is material-related
        if self.is_materials_science_query(task):
            query_material_command = CompletionModelFunction(
                name="query_materials_project",
                description="Query material properties using the Materials Project API",
                parameters={
                    "type": "object",
                    "properties": {
                        "material_id": {
                            "type": "string",
                            "description": "The material ID or formula",
                        },
                        "properties": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of properties to query (e.g., ['band_gap', 'density'])",
                        }
                    },
                    "required": ["material_id", "properties"],
                },
            )
            commands.append(query_material_command)

        system_prompt, response_prefill = self.build_system_prompt(
            ai_profile=ai_profile,
            ai_directives=ai_directives,
            commands=commands,
            include_os_info=include_os_info,
        )

        # Add materials science context if relevant
        if self.is_materials_science_query(task):
            materials_science_context = self.generate_materials_science_context()
            system_prompt += f"\n\n{materials_science_context}"

        final_instruction_msg = ChatMessage.user(self.config.choose_action_instruction)

        return ChatPrompt(
            messages=[
                ChatMessage.system(system_prompt),
                ChatMessage.user(f'"""{task}"""'),
                *messages,
                final_instruction_msg,
            ],
            prefill_response=response_prefill,
            functions=commands if self.config.use_functions_api else [],
        )

    def generate_materials_science_context(self) -> str:
        """Generate context for materials science queries."""
        return """
        ## Materials Science Capabilities
        You have access to the Materials Project database for materials science queries. When dealing with materials science tasks:
        1. Use the 'query_materials_project' command to access the Materials Project database.
        2. Identify specific materials by their formula or Materials Project ID.
        3. Request relevant properties such as band gap, density, or crystal structure.
        4. Interpret the results in the context of the user's query.
        5. If the Materials Project database doesn't have the information, consider suggesting alternative sources or experiments.
        6. Always provide explanations and context for the materials properties you discuss.
        """

    #Method for building system prompt
    def build_system_prompt(
        self,
        ai_profile: AIProfile,
        ai_directives: AIDirectives,
        commands: List[CompletionModelFunction],
        include_os_info: bool,
    ) -> Tuple[str, str]:
        """
        Builds the system prompt.

        Returns:
            str: The system prompt body
            str: The desired start for the LLM's response; used to steer the output
        """
        response_fmt_instruction, response_prefill = self.response_format_instruction(
            self.config.use_functions_api
        )
        
        # Add materials science capabilities to resources if enabled
        if self.config.enable_materials_science:
            ai_directives.resources.append(
                "Access to the Materials Project database for materials science queries."
            )

        # Prepare the materials science info
        materials_science_info = self.config.materials_science_info if self.config.enable_materials_science else ""

        system_prompt_parts = (
            self._generate_intro_prompt(ai_profile)
            + (self._generate_os_info() if include_os_info else [])
            + [
                self.config.body_template.format(
                    constraints=format_numbered_list(ai_directives.constraints),
                    resources=format_numbered_list(ai_directives.resources),
                    commands=self._generate_commands_list(commands),
                    best_practices=format_numbered_list(ai_directives.best_practices),
                    materials_science_info=materials_science_info  # Add this line
                )
            ]
            + [
                "## Your Task\n"
                "The user will specify a task for you to execute, in triple quotes,"
                " in the next message. Your job is to complete the task while following"
                " your directives as given above, and terminate when your task is done."
            ]
            + ["## RESPONSE FORMAT\n" + response_fmt_instruction]
        )

        # Join non-empty parts together into paragraph format
        return (
            "\n\n".join(filter(None, system_prompt_parts)).strip("\n"),
            response_prefill,
        )

    
    def _get_materials_project_version(self) -> str:
        """Retrieves the version of the Materials Project API being used.

        Returns:
            str: The version of the Materials Project API.
        """
        # Hardcoding the latest version as per the provided information
        return "v2023.11.1"
    


    def response_format_instruction(self, use_functions_api: bool) -> Tuple[str, str]:
        response_schema = self.response_schema.model_copy(deep=True)
        assert response_schema.properties
        if use_functions_api and "use_tool" in response_schema.properties:
            del response_schema.properties["use_tool"]

        # Unindent for performance
        response_format = re.sub(
            r"\n\s+",
            "\n",
            response_schema.to_typescript_object_interface(_RESPONSE_INTERFACE_NAME),
        )
        response_prefill = f'{{\n    "{list(response_schema.properties.keys())[0]}":'

        # Add context for materials science responses
        if self.materials_project_api_key:
            response_format += "\n\nFor materials science queries, ensure that the response includes:\n" \
                               "1. The material ID or formula used in the query.\n" \
                               "2. A list of requested properties (e.g., band gap, density).\n" \
                               "3. Relevant data retrieved from the Materials Project database.\n" \
                               "4. Clear explanations and context for the material properties discussed."

        return (
            (
                f"YOU MUST ALWAYS RESPOND WITH A JSON OBJECT OF THE FOLLOWING TYPE:\n"
                f"{response_format}"
                + ("\n\nYOU MUST ALSO INVOKE A TOOL!" if use_functions_api else "")
            ),
            response_prefill,
        )

    def _generate_intro_prompt(self, ai_profile: AIProfile) -> list[str]:
        """Generates the introduction part of the prompt.

        Args:
            ai_profile (AIProfile): The profile of the AI assistant.

        Returns:
            list[str]: A list of strings forming the introduction part of the prompt.
        """
        intro = [
            f"You are {ai_profile.ai_name}, {ai_profile.ai_role.rstrip('.')}.",
            "Your decisions must always be made independently without seeking "
            "user assistance. Play to your strengths as an LLM and pursue "
            "simple strategies with no legal complications.",
        ]

        if self.materials_project_api_key:
            intro.extend([
                "You have specialized knowledge in materials science and access to the Materials Project database.",
                "For materials science queries, you can provide detailed information about material properties, "
                "structures, and characteristics using data from the Materials Project.",
                "When addressing materials science topics, always strive to give accurate, scientifically sound "
                "explanations and interpretations of the data."
            ])

        return intro

    def _generate_os_info(self) -> list[str]:
        """Generates the OS information part of the prompt.

        Params:
            config (Config): The configuration object.

        Returns:
            str: The OS information part of the prompt.
        """
        os_name = platform.system()
        os_info = (
            platform.platform(terse=True)
            if os_name != "Linux"
            else distro.name(pretty=True)
        )
        return [f"The OS you are running on is: {os_info}"]
    
    def materials_science_info(self) -> list[str]:
        """Generates information about the materials science environment.

        Returns:
            list[str]: Information about the materials science capabilities.
        """
        if not self.materials_project_api_key:
            return []

        return [
            "Materials Science Environment:",
            "- You have access to the Materials Project database.",
            "- You can query material properties, structures, and other relevant data.",
            "- Use the 'query_materials_project' command for materials science queries.",
            f"- Materials Project API version: {self._get_materials_project_version()}",
            "- Always interpret materials data in the context of the query and provide scientific explanations."
        ]

    def _generate_commands_list(self, commands: List[CompletionModelFunction]) -> str:
        """Lists the commands available to the agent.

        Params:
            commands: The list of commands available to the agent.

        Returns:
            str: A string containing a numbered list of commands.
        """
        try:
            command_lines = [cmd.fmt_line() for cmd in commands]
            
            # Add Materials Project command if API key is available
            if self.materials_project_api_key:
                materials_project_command = "query_materials_project: Query the Materials Project database for materials information"
                command_lines.append(materials_project_command)
            
            return format_numbered_list(command_lines)
        except AttributeError:
            self.logger.warning(f"Formatting commands failed. {commands}")
            raise

    def parse_response_content(
        self,
        response: AssistantChatMessage,
    ) -> OneShotAgentActionProposal:
        if not response.content:
            raise InvalidAgentResponseError("Assistant response has no text content")

        self.logger.debug(
            "LLM response content:"
            + (
                f"\n{response.content}"
                if "\n" in response.content
                else f" '{response.content}'"
            )
        )
        assistant_reply_dict = extract_dict_from_json(response.content)
        self.logger.debug(
            "Parsing object extracted from LLM response:\n"
            f"{json.dumps(assistant_reply_dict, indent=4)}"
        )
        
        if self.config.use_functions_api:
            if not response.tool_calls:
                raise InvalidAgentResponseError("Assistant did not use a tool")
            assistant_reply_dict["use_tool"] = response.tool_calls[0].function

        # Handle materials science-related responses
        if self.is_materials_science_response(assistant_reply_dict):
            self.enhance_materials_science_response(assistant_reply_dict)

        parsed_response = OneShotAgentActionProposal.model_validate(
            assistant_reply_dict
        )
        parsed_response.raw_message = response.copy()
        return parsed_response

    def is_materials_science_response(self, response_dict: Dict[str, Any]) -> bool:
        """Check if the response is related to materials science."""
        if "use_tool" in response_dict and response_dict["use_tool"].get("name") == "query_materials_project":
            return True
        if "thoughts" in response_dict and any(keyword in response_dict["thoughts"].get("text", "").lower() for keyword in self.material_keywords):
            return True
        return False

    def enhance_materials_science_response(self, response_dict: Dict[str, Any]) -> None:
        """Enhance the materials science response with additional context or processing."""
        if "thoughts" not in response_dict:
            response_dict["thoughts"] = {}

        # Ensure that the plan is a list
        current_plan = response_dict["thoughts"].get("plan", [])
        if isinstance(current_plan, str):
            current_plan = [current_plan]  # Convert string to a single-item list
        elif not isinstance(current_plan, list):
            current_plan = []  # Initialize an empty list if it's neither a string nor a list

        # Append new steps to the plan
        current_plan.extend([
            "Query the Materials Project database",
            "Analyze the retrieved data",
            "Interpret the results in the context of the user's query",
            "Provide a clear explanation of the findings"
        ])

        # Update the plan in the response dictionary
        response_dict["thoughts"]["plan"] = current_plan
