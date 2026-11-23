"""The city registry is depended on by model filenames and API paths."""

import pytest

from wepr import cities

# Generous bounding box for the Iberian peninsula plus the Balearics.
LAT_RANGE = (35.5, 44.5)
LON_RANGE = (-10.0, 4.5)


def test_twenty_cities():
    assert len(cities.CITIES) == 20


def test_slugs_are_unique():
    slugs = [c.slug for c in cities.CITIES]
    assert len(slugs) == len(set(slugs))


def test_slugs_are_url_safe():
    for c in cities.CITIES:
        assert c.slug.replace("-", "").isalnum(), c.slug
        assert c.slug == c.slug.lower(), c.slug


def test_coordinates_are_in_iberia():
    for c in cities.CITIES:
        assert LAT_RANGE[0] <= c.lat <= LAT_RANGE[1], c
        assert LON_RANGE[0] <= c.lon <= LON_RANGE[1], c


def test_countries_are_iso_alpha2():
    for c in cities.CITIES:
        assert c.country in {"ES", "PT"}, c


def test_every_region_is_populated():
    for region in cities.regions():
        assert any(c.region == region for c in cities.CITIES)


def test_validation_cities_are_registered():
    for slug in cities.VALIDATION_SLUGS:
        assert slug in cities.BY_SLUG


def test_unknown_slug_raises_with_a_helpful_message():
    with pytest.raises(KeyError, match="unknown city"):
        cities.get("atlantis")


def test_city_is_immutable():
    """Slugs appear in model filenames, so a city must not be mutated in place."""
    with pytest.raises(Exception):
        cities.CITIES[0].slug = "changed"
