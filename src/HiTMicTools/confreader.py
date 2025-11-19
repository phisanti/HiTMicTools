import yaml
import os
from typing import Any, Dict


class ConfReader:
    """
    Load yaml config file using DictWithAttributeAccess object_hook.
    ConfLoader(conf_name).opt attribute is the result of loading yaml config file.
    """

    class DictWithAttributeAccess(dict):
        """
        This inner class makes dict to be accessed same as class attribute.
        For example, you can use opt.key instead of the opt['key'].
        """

        def __getattr__(self, key: str) -> Any:
            """Return the stored value when an attribute-style lookup is performed."""
            return self[key]

        def __setattr__(self, key: str, value: Any) -> None:
            """Store an attribute-style assignment back into the underlying dict."""
            self[key] = value

    def __init__(self, conf_name: str) -> None:
        """
        Initialize the configuration reader.

        Args:
            conf_name: Absolute or relative path to the YAML configuration file.
        """
        self.conf_name = conf_name
        self.opt = self.__get_opt()

    def __load_conf(self) -> Dict[str, Any]:
        """
        Load the YAML file from disk and return it as a standard dictionary.

        Returns:
            Dict[str, Any]: Parsed YAML contents.

        Raises:
            AssertionError: If the configuration file does not exist.
        """
        assert os.path.exists(self.conf_name), f"File {self.conf_name} not found"
        with open(self.conf_name, "r") as conf:
            opt = yaml.safe_load(conf)
        return opt

    def __get_opt(self) -> DictWithAttributeAccess:
        """
        Convert the parsed configuration dictionary into the attribute-accessible wrapper.

        Returns:
            DictWithAttributeAccess: Configuration object with attribute-style access.
        """
        opt = self.__load_conf()
        opt = self.DictWithAttributeAccess(opt)
        return opt

    def pretty_print(
        self,
        d: Dict[str, Any],
        title: str = "Settings",
        indent: int = 0,
        direct_print: bool = False,
    ) -> str:
        """
        Format configuration entries as a human readable string.

        Args:
            d: Dictionary to format.
            title: Heading shown at the top of the output.
            indent: Number of spaces to prepend when nesting dictionaries.
            direct_print: If True the string is printed immediately, otherwise returned.

        Returns:
            str: Formatted configuration string unless direct_print=True.
        """
        output = f"{title}:\n"
        for key, value in d.items():
            output += " " * indent + str(key) + ": "
            if isinstance(value, dict):
                output += "\n" + self.pretty_print(value, indent=indent + 2)
            else:
                output += str(value) + "\n"
        if direct_print:
            print(output)
        else:
            return output
