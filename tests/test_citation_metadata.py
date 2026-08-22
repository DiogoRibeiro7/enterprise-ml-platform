"""The citation files must agree with each other and with the package.

``.zenodo.json`` and ``CITATION.cff`` describe the software to people who will
never read the code: a Zenodo record is permanent and citable, and GitHub
renders the CFF file in the sidebar. Both were written from the project's
original prospectus and claimed a "comprehensive, production-ready" platform
long after the README stopped doing so. Nothing was checking that the three
descriptions matched.
"""

from __future__ import annotations

import json
import pathlib

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ZENODO = REPO_ROOT / ".zenodo.json"
CITATION = REPO_ROOT / "CITATION.cff"
PYPROJECT = REPO_ROOT / "pyproject.toml"

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - older interpreters
    tomllib = pytest.importorskip("tomli")

#: Fields Zenodo's legacy deposition schema accepts. Anything else is dropped
#: silently on ingest, so a typo would go unnoticed until the record is public.
ZENODO_FIELDS = frozenset(
    {
        "upload_type",
        "publication_type",
        "image_type",
        "publication_date",
        "title",
        "creators",
        "description",
        "access_right",
        "license",
        "embargo_date",
        "access_conditions",
        "doi",
        "prereserve_doi",
        "keywords",
        "notes",
        "related_identifiers",
        "contributors",
        "references",
        "communities",
        "grants",
        "journal_title",
        "journal_volume",
        "journal_issue",
        "journal_pages",
        "conference_title",
        "conference_acronym",
        "conference_dates",
        "conference_place",
        "conference_url",
        "conference_session",
        "conference_session_part",
        "imprint_publisher",
        "imprint_isbn",
        "imprint_place",
        "partof_title",
        "partof_pages",
        "thesis_supervisors",
        "thesis_university",
        "subjects",
        "version",
        "language",
        "locations",
        "dates",
        "method",
    }
)

#: Creators take no ``type``; that belongs to contributors. Zenodo rejects the
#: field rather than ignoring it.
CREATOR_FIELDS = frozenset({"name", "affiliation", "orcid", "gnd"})

CONTRIBUTOR_TYPES = frozenset(
    {
        "ContactPerson",
        "DataCollector",
        "DataCurator",
        "DataManager",
        "Distributor",
        "Editor",
        "Funder",
        "HostingInstitution",
        "Producer",
        "ProjectLeader",
        "ProjectManager",
        "ProjectMember",
        "RegistrationAgency",
        "RegistrationAuthority",
        "RelatedPerson",
        "Researcher",
        "ResearchGroup",
        "RightsHolder",
        "Supervisor",
        "Sponsor",
        "WorkPackageLeader",
        "Other",
    }
)

#: Claims the README stopped making. A citation record outlives a README, so
#: it is the last place they should survive.
RETIRED_CLAIMS = (
    "production-ready",
    "production-oriented",
    "comprehensive",
    "multi-cloud",
    "enterprise environments",
)


@pytest.fixture(scope="module")
def zenodo() -> dict:
    return json.loads(ZENODO.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def citation() -> dict:
    return yaml.safe_load(CITATION.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def project() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]


# ----------------------------------------------------------------------
# Zenodo record shape
# ----------------------------------------------------------------------
def test_zenodo_uses_only_fields_zenodo_accepts(zenodo: dict) -> None:
    assert not set(zenodo) - ZENODO_FIELDS


def test_zenodo_creators_carry_no_contributor_fields(zenodo: dict) -> None:
    for creator in zenodo["creators"]:
        assert not set(creator) - CREATOR_FIELDS, creator


def test_zenodo_contributor_types_are_from_the_vocabulary(zenodo: dict) -> None:
    for contributor in zenodo["contributors"]:
        assert contributor["type"] in CONTRIBUTOR_TYPES


def test_zenodo_declares_an_open_software_deposit(zenodo: dict) -> None:
    assert zenodo["upload_type"] == "software"
    assert zenodo["access_right"] == "open"


def test_zenodo_does_not_pin_what_the_release_supplies(zenodo: dict) -> None:
    """Zenodo fills these from the GitHub release; pinning them freezes them.

    The first version of this file hardcoded a publication date, which was
    already wrong the day after it was written.
    """
    assert "version" not in zenodo
    assert "publication_date" not in zenodo


def test_zenodo_description_html_is_balanced(zenodo: dict) -> None:
    from html.parser import HTMLParser

    class Balanced(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.stack: list[str] = []
            self.errors: list[str] = []

        def handle_starttag(self, tag: str, attrs: list) -> None:
            if tag not in {"br", "img", "hr"}:
                self.stack.append(tag)

        def handle_endtag(self, tag: str) -> None:
            if not self.stack or self.stack[-1] != tag:
                self.errors.append(f"unbalanced </{tag}>")
            else:
                self.stack.pop()

    parser = Balanced()
    parser.feed(zenodo["description"])
    assert not parser.errors, parser.errors
    assert not parser.stack, f"unclosed: {parser.stack}"


# ----------------------------------------------------------------------
# The three files agree
# ----------------------------------------------------------------------
def test_the_same_person_is_credited_everywhere(zenodo: dict, citation: dict) -> None:
    author = citation["authors"][0]
    creator = zenodo["creators"][0]

    assert creator["name"] == f"{author['family-names']}, {author['given-names']}"
    assert creator["affiliation"] == author["affiliation"]
    assert author["orcid"].endswith(creator["orcid"])


def test_licences_agree(zenodo: dict, citation: dict, project: dict) -> None:
    declared = project.get("license")
    declared = declared.get("text") if isinstance(declared, dict) else declared

    assert zenodo["license"].lower() == "mit"
    assert citation["license"] == "MIT"
    assert declared == "MIT"


def test_keywords_agree(zenodo: dict, citation: dict) -> None:
    assert sorted(zenodo["keywords"]) == sorted(citation["keywords"])


def test_repository_urls_use_the_canonical_owner_casing(citation: dict) -> None:
    """A lower-cased owner only works because GitHub redirects."""
    for field in ("repository-code", "repository-artifact"):
        url = citation.get(field)
        if url:
            assert "DiogoRibeiro7" in url, f"{field}: {url}"


# ----------------------------------------------------------------------
# Neither file outlives the claims the README dropped
# ----------------------------------------------------------------------
@pytest.mark.parametrize("claim", RETIRED_CLAIMS)
def test_zenodo_description_makes_no_retired_claim(zenodo: dict, claim: str) -> None:
    assert claim.lower() not in zenodo["description"].lower()


@pytest.mark.parametrize("claim", RETIRED_CLAIMS)
def test_citation_abstract_makes_no_retired_claim(citation: dict, claim: str) -> None:
    assert claim.lower() not in citation["abstract"].lower()


def test_citation_abstract_says_what_is_scaffolding(citation: dict) -> None:
    """The abstract has to carry the caveat, not only the capabilities."""
    assert "scaffolding" in citation["abstract"].lower()


def test_zenodo_description_states_its_limitations(zenodo: dict) -> None:
    assert "limitations" in zenodo["description"].lower()
