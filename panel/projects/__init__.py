"""Project registry.

Adding a project to the panel means writing one module in this package that
exposes a `PROJECT` object, then listing it in REGISTRY below. The tab bar,
the run form, the frame/GIF viewer and the scheduler are all driven off these
records, so no UI work is needed for a new project.
"""
from projects.base import Param, Project, ProjectError  # noqa: F401
from projects import nexrad, wazuh

REGISTRY = [
    nexrad.PROJECT,
    wazuh.PROJECT,
]

BY_ID = {p.id: p for p in REGISTRY}


def get(project_id):
    try:
        return BY_ID[project_id]
    except KeyError:
        raise ProjectError("unknown project %r" % project_id)
