from .config import PromptConfig


class PromptAdapter:
    def __init__(self, config: PromptConfig):
        self.config = config

    def render(self, item):
        template = self.config.query_template if item.role == "query" else self.config.document_template
        rendered = template.format(instruction=item.instruction, text=item.text)
        return rendered.strip() if self.config.strip_rendered else rendered

    def diagnostics(self):
        return {"instruction_preserved": "{instruction}" in self.config.query_template,
                "template_version": self.config.version}
