"""
Nornir task runner for Atlas network automation.
Renders and deploys Arista EOS configurations via Jinja2 templates.
"""

import argparse
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from nornir import InitNornir
from nornir.core.task import Result, Task
from nornir_netmiko.tasks import netmiko_send_command, netmiko_send_config
from nornir_utils.plugins.functions import print_result

TEMPLATE_DIR = Path(__file__).parent / "templates"
CONFIG_FILE = Path(__file__).parent / "config.yaml"
COLLECT_DIR = Path(__file__).parent / "collected"

COLLECT_COMMANDS = {
    "arp":        "show ip arp",
    "routes":     "show ip route",
    "ospf":       "show ip ospf neighbor",
    "interfaces": "show interfaces status",
    "bgp":        "show ip bgp summary",
    "lldp":       "show lldp neighbors",
    "mac":        "show mac address-table",
}


def collect_data(task: Task, commands: list[str]) -> Result:
    """Run show commands and save output to collected/<host>/<command>.txt"""
    host_dir = COLLECT_DIR / task.host.name
    host_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}
    for cmd in commands:
        result = task.run(task=netmiko_send_command, command_string=cmd)
        output = result.result
        outputs[cmd] = output
        filename = cmd.replace(" ", "_") + ".txt"
        (host_dir / filename).write_text(output)

    summary = "\n\n".join(f"--- {cmd} ---\n{out}" for cmd, out in outputs.items())
    return Result(host=task.host, result=summary)


def render_config(task: Task) -> Result:
    """Render device config from Jinja2 template using host inventory data."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("arista_eos.j2")

    rendered = template.render(
        inventory_hostname=task.host.name,
        **task.host.data,
    )
    task.host["rendered_config"] = rendered
    return Result(host=task.host, result=rendered)


def deploy_config(task: Task) -> Result:
    """Push rendered config lines to device via Netmiko."""
    config_lines = task.host.get("rendered_config", "").splitlines()
    config_lines = [line for line in config_lines if line.strip() and line.strip() != "!"]

    result = task.run(
        task=netmiko_send_config,
        config_commands=config_lines,
    )
    return Result(host=task.host, result=result)


def render_only(task: Task) -> Result:
    """Render config and save to file without deploying."""
    render_result = task.run(task=render_config)
    rendered = task.host.get("rendered_config", "")

    output_dir = Path(__file__).parent / "rendered"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{task.host.name}.cfg"
    output_file.write_text(rendered)

    return Result(host=task.host, result=f"Config saved to {output_file}")


def main():
    collect_choices = list(COLLECT_COMMANDS.keys()) + ["all"]

    parser = argparse.ArgumentParser(description="Nornir config manager for Arista EOS")
    parser.add_argument("--generate", action="store_true", help="Render configs to nornir/rendered/")
    parser.add_argument("--push", action="store_true", help="Push configs to devices")
    parser.add_argument(
        "--collect",
        nargs="+",
        choices=collect_choices,
        metavar=f"{{{','.join(collect_choices)}}}",
        help="Collect data from devices (e.g. --collect arp routes ospf, or --collect all)",
    )
    parser.add_argument("--host", help="Target a single host (e.g. arista_1)")
    args = parser.parse_args()

    if not args.generate and not args.push and not args.collect:
        parser.print_help()
        return

    nr = InitNornir(config_file=str(CONFIG_FILE))

    if args.host:
        nr = nr.filter(name=args.host)

    if args.generate:
        print("=== Rendering configurations ===")
        result = nr.run(task=render_only)
        print_result(result)

    if args.push:
        if args.generate:
            print("\n=== Pushing configurations ===")
            result = nr.run(task=deploy_config)
        else:
            print("=== Rendering + pushing configurations ===")
            nr.run(task=render_config)
            result = nr.run(task=deploy_config)
        print_result(result)

    if args.collect:
        selected = list(COLLECT_COMMANDS.keys()) if "all" in args.collect else args.collect
        commands = [COLLECT_COMMANDS[k] for k in selected]
        print(f"\n=== Collecting: {', '.join(selected)} ===")
        result = nr.run(task=collect_data, commands=commands)
        print_result(result)
        print(f"\nOutput saved to {COLLECT_DIR}/")


if __name__ == "__main__":
    main()
