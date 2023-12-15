import distrax as dx
import yaml


def add_representer(type: type[dx.DistributionLike]) -> None:
    """Add a simple representer for a given distribution class"""

    def representer(dumper: yaml.Dumper, data: dx.DistributionLike):
        return dumper.represent_str(data.name)

    yaml.representer.SafeRepresenter.add_representer(type, representer)
    yaml.add_representer(type, representer)


def all_distributions() -> set[dx.DistributionLike]:
    """Find all the classes in the inheritance graph of dx.Distribution"""
    subdists = set()
    work = [dx.Distribution]

    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subdists:
                subdists.add(child)
                work.append(child)

    return subdists


def add_distrax_representers() -> None:
    distributions = all_distributions()
    for d in distributions:
        add_representer(d)
