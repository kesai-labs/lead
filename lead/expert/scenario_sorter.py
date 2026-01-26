"""Efficient scenario sorter that uses XML file ordering for scenario prioritization."""

import logging
import xml.etree.ElementTree as ET

import carla
import numpy as np
from beartype import beartype
from srunner.scenariomanager.carla_data_provider import (
    ActiveScenario,
    CarlaDataProvider,
)

LOG = logging.getLogger(__name__)


class XMLScenarioInfo:
    """Stores information about a scenario from the XML file."""

    def __init__(self, index: int, name: str, scenario_type: str, trigger_points: list):
        self.index = index
        self.name = name
        self.scenario_type = scenario_type
        self.trigger_points = trigger_points  # List of carla.Location objects


class ScenarioSorter:
    """Sorts active scenarios by their order in the XML file.

    This class uses the scenario order defined in the route XML file to sort
    active scenarios, which is more reliable than distance-based sorting especially
    for routes with loops.
    """

    @beartype
    def __init__(self):
        """Initialize the scenario sorter."""
        self._xml_scenarios: list[XMLScenarioInfo] = []  # List of XML scenario info
        self._xml_loaded = False

    def _load_xml_scenario_order(self):
        """Load scenario information from the XML file specified in CarlaDataProvider."""
        xml_path = CarlaDataProvider.get_route_xml_path()
        if xml_path is None:
            LOG.warning(
                "No XML path set in CarlaDataProvider. Cannot load scenario order."
            )
            return

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Find the scenarios section
            for route in root.iter("route"):
                scenarios = route.find("scenarios")
                if scenarios is not None:
                    for idx, scenario in enumerate(scenarios.iter("scenario")):
                        scenario_name = scenario.attrib.get("name")
                        scenario_type = scenario.attrib.get("type")

                        # Extract trigger points
                        trigger_points = []
                        for trigger_elem in scenario.findall("trigger_point"):
                            try:
                                x = float(trigger_elem.attrib.get("x", 0))
                                y = float(trigger_elem.attrib.get("y", 0))
                                z = float(trigger_elem.attrib.get("z", 0))
                                trigger_points.append(carla.Location(x=x, y=y, z=z))
                            except (ValueError, TypeError) as e:
                                LOG.warning(
                                    f"Failed to parse trigger point for {scenario_name}: {e}"
                                )

                        xml_info = XMLScenarioInfo(
                            index=idx,
                            name=scenario_name,
                            scenario_type=scenario_type,
                            trigger_points=trigger_points,
                        )
                        self._xml_scenarios.append(xml_info)

                    break  # Only process the first route (assumes single route per file)

            self._xml_loaded = True
            LOG.info(
                f"Loaded {len(self._xml_scenarios)} scenarios from XML: {xml_path}"
            )
        except Exception as e:
            LOG.error(f"Failed to load XML scenario order from {xml_path}: {e}")
            self._xml_loaded = False

    def _remove_ended_scenarios(self) -> None:
        """Remove scenarios that have ended (actors are no longer alive).

        This cleans up scenarios where the first_actor or last_actor is no longer alive,
        which typically indicates the scenario has completed or timed out.
        """
        active_scenarios = CarlaDataProvider.active_scenarios.copy()

        for scenario in active_scenarios:
            should_remove = False

            # Check if first actor is dead
            if scenario.first_actor is not None and not scenario.first_actor.is_alive:
                should_remove = True

            # Check if last actor exists and is dead
            if (
                not should_remove
                and scenario.last_actor is not None
                and not scenario.last_actor.is_alive
            ):
                should_remove = True

            if should_remove:
                CarlaDataProvider.remove_scenario(scenario)

    @beartype
    def _match_scenario_to_xml(
        self, scenario: ActiveScenario
    ) -> XMLScenarioInfo | None:
        """Match an active scenario to an XML scenario by comparing all available attributes.

        Args:
            scenario: The active scenario to match.

        Returns:
            The matched XMLScenarioInfo or None if no match found.
        """
        best_match = None
        best_score = 0

        # Get scenario attributes
        scenario_name = scenario.name
        scenario_trigger_loc = (
            scenario.trigger_location if hasattr(scenario, "trigger_location") else None
        )

        # Try to match against each XML scenario
        for xml_info in self._xml_scenarios:
            score = 0

            # 1. Exact name match (highest priority)
            if xml_info.name == scenario_name:
                score += 1000

            # 2. Type match (scenario name might be just the type)
            if xml_info.scenario_type == scenario_name:
                score += 100

            # 3. Name contains type or type contains name
            if xml_info.name and scenario_name:
                if scenario_name in xml_info.name or xml_info.name.startswith(
                    scenario_name
                ):
                    score += 50

            # 4. Trigger location match (if available)
            if scenario_trigger_loc and xml_info.trigger_points:
                # Check if any XML trigger point is close to the scenario trigger location
                for xml_trigger in xml_info.trigger_points:
                    distance = scenario_trigger_loc.distance(xml_trigger)
                    if distance < 5.0:  # Within 5 meters
                        score += 200
                        break
                    elif distance < 20.0:  # Within 20 meters
                        score += 50
                        break

            # Update best match if this score is higher
            if score > best_score:
                best_score = score
                best_match = xml_info

        # Only return a match if we have some confidence
        if best_score > 0:
            LOG.debug(
                f"Matched scenario '{scenario_name}' to XML scenario '{best_match.name}' (score: {best_score})"
            )
            return best_match
        else:
            LOG.warning(
                f"Could not match scenario '{scenario_name}' to any XML scenario"
            )
            return None

    @beartype
    def sort_scenarios(self) -> None:
        """Sort active scenarios by their order in the XML file.

        This method first matches each active scenario to its corresponding XML scenario
        by comparing all available attributes, then sorts by XML order.

        Also removes scenarios that have ended (actors no longer alive).
        """
        # First, remove any scenarios that have ended
        self._remove_ended_scenarios()

        active_scenarios = CarlaDataProvider.active_scenarios

        if not active_scenarios:
            return

        # Load XML order if not already loaded
        if not self._xml_loaded:
            self._load_xml_scenario_order()

        # If we couldn't load XML scenarios, fall back to keeping current order
        if not self._xml_scenarios:
            LOG.warning(
                "No XML scenario data available. Keeping current scenario order."
            )
            return

        # Match each active scenario to XML and create a list of (scenario, xml_index) pairs
        scenario_matches = []
        for scenario in active_scenarios:
            xml_match = self._match_scenario_to_xml(scenario)
            if xml_match:
                scenario_matches.append((scenario, xml_match.index))
            else:
                # No match found, push to end
                scenario_matches.append((scenario, 999999))

        # Sort by XML index
        scenario_matches.sort(key=lambda x: x[1])

        # Update the active scenarios list with sorted order
        CarlaDataProvider.active_scenarios = [
            scenario for scenario, _ in scenario_matches
        ]

        LOG.debug(f"Sorted {len(active_scenarios)} scenarios by XML order")

    @beartype
    def _get_scenario_location(self, scenario: ActiveScenario) -> carla.Location | None:
        """Extract location from scenario (first_actor or trigger_location).

        Args:
            scenario: The scenario object.

        Returns:
            Location or None if not available.
        """
        if scenario.first_actor is not None and scenario.first_actor.is_alive:
            try:
                return scenario.first_actor.get_location()
            except:
                pass

        if (
            hasattr(scenario, "trigger_location")
            and scenario.trigger_location is not None
        ):
            return scenario.trigger_location

        return None

    @beartype
    def _sort_by_euclidean_distance(self, ego_location: carla.Location) -> None:
        """Fallback sorting using simple Euclidean distance.

        Args:
            ego_location: Current ego vehicle location.
        """
        active_scenarios = CarlaDataProvider.active_scenarios
        distances = []

        for scenario in active_scenarios:
            scenario_location = self._get_scenario_location(scenario)
            if scenario_location is not None:
                distance = ego_location.distance(scenario_location)
            else:
                distance = float("inf")
            distances.append(distance)

        indices = np.argsort(distances)
        CarlaDataProvider.active_scenarios = [active_scenarios[i] for i in indices]
