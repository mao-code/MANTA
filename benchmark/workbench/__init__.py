"""WorkBench benchmark adapter.

Source repo : https://github.com/olly-styles/WorkBench
Paper       : https://arxiv.org/abs/2405.00823

WorkBench evaluates workplace agents on realistic tool-using tasks. Each task
is a natural-language request ("query" in the upstream code) paired with a
ground-truth list of tool calls ("answer"). Correctness is determined by the
resulting state change across sandbox databases rather than free-form text.
"""

from __future__ import annotations

import ast
import csv
import logging
import re
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from loguru import logger
except Exception:  # pragma: no cover - optional dependency fallback
    logger = logging.getLogger(__name__)

from answer_utils import has_substantive_answer
from benchmark.base import BenchmarkEvaluation, BenchmarkTask
from MAS import MASRunner, MASRunResult
from MAS.artifacts import collect_lineage_tool_records

RAW_BASE_URL = "https://raw.githubusercontent.com/olly-styles/WorkBench/main/"
HARDCODED_CURRENT_TIME = pd.to_datetime("2023-11-30T00:00:00")

CORE_DOMAINS = [
    "email",
    "calendar",
    "analytics",
    "project_management",
    "customer_relationship_manager",
]

# Upstream multi-domain tasks label the CRM domain as "crm", while the
# single-domain files, TOOL_SPECS, and SIDE_EFFECT_TOOLS use the full
# "customer_relationship_manager" name. Normalize the shorthand so
# tool_selection="domains" still exposes the CRM tools those tasks need.
DOMAIN_ALIASES = {"crm": "customer_relationship_manager"}

QUERY_FILE_NAMES = {
    "analytics": "analytics_tasks_and_outcomes.csv",
    "calendar": "calendar_tasks_and_outcomes.csv",
    "customer_relationship_manager": "customer_relationship_manager_tasks_and_outcomes.csv",
    "email": "email_tasks_and_outcomes.csv",
    "multi_domain": "multi_domain_tasks_and_outcomes.csv",
    "project_management": "project_management_tasks_and_outcomes.csv",
}

DATA_FILES = [
    "data/processed/analytics_data.csv",
    "data/processed/calendar_events.csv",
    "data/processed/customer_relationship_manager_data.csv",
    "data/processed/emails.csv",
    "data/processed/project_tasks.csv",
    "data/raw/email_addresses.csv",
]

SIDE_EFFECT_TOOLS = {
    "calendar.create_event",
    "calendar.delete_event",
    "calendar.update_event",
    "email.send_email",
    "email.delete_email",
    "email.forward_email",
    "email.reply_email",
    "analytics.create_plot",
    "project_management.create_task",
    "project_management.delete_task",
    "project_management.update_task",
    "customer_relationship_manager.update_customer",
    "customer_relationship_manager.add_customer",
    "customer_relationship_manager.delete_customer",
}

FIELDS_NOT_TO_LOWER = {"status", "list_name", "board"}


class WorkBenchSandbox:
    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.reset_state()

    def reset_state(self) -> None:
        self.calendar_events = pd.read_csv(
            self.data_root / "data/processed/calendar_events.csv", dtype=str
        )
        self.emails = pd.read_csv(self.data_root / "data/processed/emails.csv", dtype=str)
        self.analytics_data = pd.read_csv(
            self.data_root / "data/processed/analytics_data.csv", dtype=str
        )
        self.analytics_data["user_engaged"] = self.analytics_data["user_engaged"] == "True"
        self.plots_data = pd.DataFrame(columns=["file_path"])
        self.project_tasks = pd.read_csv(
            self.data_root / "data/processed/project_tasks.csv", dtype=str
        )
        self.crm_data = pd.read_csv(
            self.data_root / "data/processed/customer_relationship_manager_data.csv", dtype=str
        )
        self.company_directory = pd.read_csv(
            self.data_root / "data/raw/email_addresses.csv",
            header=None,
            names=["email_address"],
            dtype=str,
        )

    def fork(self) -> WorkBenchSandbox:
        """Return an independent copy of the current state (used for dry-run checks)."""

        copy = object.__new__(WorkBenchSandbox)
        copy.data_root = self.data_root
        for name, value in vars(self).items():
            if isinstance(value, pd.DataFrame):
                setattr(copy, name, value.copy())
        return copy

    def snapshot(self) -> dict[str, pd.DataFrame]:
        return {
            "calendar": self.calendar_events.copy(),
            "email": self.emails.copy(),
            "analytics": self.plots_data.copy(),
            "project_management": self.project_tasks.copy(),
            "customer_relationship_manager": self.crm_data.copy(),
        }

    def invoke(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        method = getattr(self, "_tool_" + tool_name.replace(".", "__"), None)
        if method is None:
            raise ValueError(f"Unknown WorkBench tool: {tool_name}")
        return method(**arguments)

    def _tool_calendar__get_event_information_by_id(
        self, event_id: str | None = None, field: str | None = None
    ) -> Any:
        if not event_id:
            return "Event ID not provided."
        if not field:
            return "Field not provided."
        event = self.calendar_events[self.calendar_events["event_id"] == str(event_id)].to_dict(
            orient="records"
        )
        if not event:
            return "Event not found."
        return {field: event[0][field]} if field in event[0] else "Field not found."

    def _tool_calendar__search_events(
        self,
        query: str = "",
        time_min: str | None = None,
        time_max: str | None = None,
    ) -> Any:
        events = self.calendar_events[
            self.calendar_events["event_name"].str.contains(query, case=False, na=False)
            | self.calendar_events["participant_email"].str.contains(query, case=False, na=False)
        ].to_dict(orient="records")
        if time_min:
            events = [
                event
                for event in events
                if pd.Timestamp(event["event_start"]) >= pd.Timestamp(time_min)
            ]
        if time_max:
            events = [
                event
                for event in events
                if pd.Timestamp(event["event_start"]) <= pd.Timestamp(time_max)
            ]
        return events[:5] if events else "No events found."

    def _tool_calendar__create_event(
        self,
        event_name: str | None = None,
        participant_email: str | None = None,
        event_start: str | None = None,
        duration: str | None = None,
    ) -> str:
        if not event_name:
            return "Event name not provided."
        if not participant_email:
            return "Participant email not provided."
        if not event_start:
            return "Event start not provided."
        if not duration:
            return "Event duration not provided."
        participant_email = str(participant_email).lower()
        duration = _normalize_duration_value(duration)
        event_id = str(int(self.calendar_events["event_id"].max()) + 1).zfill(8)
        new_event = pd.DataFrame(
            {
                "event_id": [event_id],
                "event_name": [event_name],
                "participant_email": [participant_email],
                "event_start": [event_start],
                "duration": [str(duration)],
            }
        )
        self.calendar_events = pd.concat([self.calendar_events, new_event], ignore_index=True)
        return event_id

    def _tool_calendar__delete_event(self, event_id: str | None = None) -> str:
        if not event_id:
            return "Event ID not provided."
        if event_id in self.calendar_events["event_id"].values:
            self.calendar_events = self.calendar_events[
                self.calendar_events["event_id"] != event_id
            ]
            return "Event deleted successfully."
        return "Event not found."

    def _tool_calendar__update_event(
        self, event_id: str | None = None, field: str | None = None, new_value: str | None = None
    ) -> str:
        if not event_id or not field or new_value is None:
            return "Event ID, field, or new value not provided."
        if event_id in self.calendar_events["event_id"].values:
            if field == "participant_email":
                new_value = str(new_value).lower()
            elif field == "duration":
                new_value = _normalize_duration_value(new_value)
            self.calendar_events.loc[self.calendar_events["event_id"] == event_id, field] = (
                new_value
            )
            return "Event updated successfully."
        return "Event not found."

    def _tool_email__get_email_information_by_id(
        self, email_id: str | None = None, field: str | None = None
    ) -> Any:
        if not email_id:
            return "Email ID not provided."
        if not field:
            return "Field not provided."
        email = self.emails[self.emails["email_id"] == str(email_id)].to_dict(orient="records")
        if not email:
            return "Email not found."
        return {field: email[0][field]} if field in email[0] else "Field not found."

    def _tool_email__search_emails(
        self, query: str = "", date_min: str | None = None, date_max: str | None = None
    ) -> Any:
        query_words = str(query).lower().split()

        def filter_email(row: pd.Series) -> bool:
            combined = f"{row['subject']} {row['body']} {row['sender/recipient']}".lower()
            return all(word in combined for word in query_words)

        filtered = self.emails.apply(filter_email, axis=1)
        emails = (
            self.emails[filtered]
            .sort_values("sent_datetime", ascending=False)
            .to_dict(orient="records")
        )
        if date_min:
            emails = [
                email
                for email in emails
                if pd.Timestamp(email["sent_datetime"]).date() >= pd.Timestamp(date_min).date()
            ]
        if date_max:
            emails = [
                email
                for email in emails
                if pd.Timestamp(email["sent_datetime"]).date() <= pd.Timestamp(date_max).date()
            ]
        return emails[:5] if emails else "No emails found."

    def _tool_email__send_email(
        self, recipient: str | None = None, subject: str | None = None, body: str | None = None
    ) -> str:
        if not recipient or not subject or not body:
            return "Recipient, subject, or body not provided."
        if "@" not in str(recipient) or "." not in str(recipient):
            return "Invalid recipient email address."
        recipient = str(recipient).lower()
        email_id = str(int(self.emails["email_id"].max()) + 1)
        self.emails.loc[len(self.emails)] = [
            email_id,
            "outbox",
            recipient,
            subject,
            HARDCODED_CURRENT_TIME,
            body,
        ]
        return "Email sent successfully."

    def _tool_email__delete_email(self, email_id: str | None = None) -> str:
        if not email_id:
            return "Email ID not provided."
        if email_id in self.emails["email_id"].values:
            self.emails = self.emails[self.emails["email_id"] != email_id]
            return "Email deleted successfully."
        return "Email not found."

    def _tool_email__forward_email(
        self, email_id: str | None = None, recipient: str | None = None
    ) -> str:
        if not email_id or not recipient:
            return "Email ID or recipient not provided."
        if email_id not in self.emails["email_id"].values:
            return "Email not found."
        if "@" not in str(recipient) or "." not in str(recipient):
            return "Invalid recipient email address."
        email = self.emails[self.emails["email_id"] == email_id].to_dict(orient="records")[0]
        result = self._tool_email__send_email(
            recipient=str(recipient).lower(),
            subject=f"FW: {email['subject']}",
            body=email["body"],
        )
        return "Email forwarded successfully." if result == "Email sent successfully." else result

    def _tool_email__reply_email(self, email_id: str | None = None, body: str | None = None) -> str:
        if not email_id or not body:
            return "Email ID or body not provided."
        if email_id not in self.emails["email_id"].values:
            return "Email not found."
        email = self.emails[self.emails["email_id"] == email_id].to_dict(orient="records")[0]
        result = self._tool_email__send_email(
            recipient=email["sender/recipient"], subject=email["subject"], body=body
        )
        return "Email replied successfully." if result == "Email sent successfully." else result

    def _tool_analytics__get_visitor_information_by_id(self, visitor_id: str | None = None) -> Any:
        if not visitor_id:
            return "Visitor ID not provided."
        visitor = self.analytics_data[self.analytics_data["visitor_id"] == str(visitor_id)].to_dict(
            orient="records"
        )
        return visitor if visitor else "Visitor not found."

    def _tool_analytics__create_plot(
        self,
        time_min: str | None = None,
        time_max: str | None = None,
        value_to_plot: str | None = None,
        plot_type: str | None = None,
    ) -> str:
        allowed_values = {
            "total_visits",
            "session_duration_seconds",
            "user_engaged",
            "visits_direct",
            "visits_referral",
            "visits_search_engine",
            "visits_social_media",
        }
        if not time_min:
            return "Start date not provided."
        if not time_max:
            return "End date not provided."
        if value_to_plot not in allowed_values:
            return (
                "Value to plot must be one of 'total_visits', 'session_duration_seconds', "
                "'user_engaged', 'direct', 'referral', 'search engine', 'social media'"
            )
        if plot_type not in {"bar", "line", "scatter", "histogram"}:
            return "Plot type must be one of 'bar', 'line', 'scatter', or 'histogram'"
        file_path = f"plots/{time_min}_{time_max}_{value_to_plot}_{plot_type}.png"
        self.plots_data.loc[len(self.plots_data)] = [file_path]
        return file_path

    def _tool_analytics__total_visits_count(
        self, time_min: str | None = None, time_max: str | None = None
    ) -> dict[str, int]:
        data = self.analytics_data
        if time_min:
            data = data[data["date_of_visit"] >= time_min]
        if time_max:
            data = data[data["date_of_visit"] <= time_max]
        return data.groupby("date_of_visit").size().to_dict()

    def _tool_analytics__engaged_users_count(
        self, time_min: str | None = None, time_max: str | None = None
    ) -> dict[str, int]:
        data = self.analytics_data.copy()
        if time_min:
            data = data[data["date_of_visit"] >= time_min]
        if time_max:
            data = data[data["date_of_visit"] <= time_max]
        data["user_engaged"] = data["user_engaged"].astype(bool).astype(int)
        return data.groupby("date_of_visit").sum(numeric_only=True)["user_engaged"].to_dict()

    def _tool_analytics__traffic_source_count(
        self,
        time_min: str | None = None,
        time_max: str | None = None,
        traffic_source: str | None = None,
    ) -> dict[str, int]:
        data = self.analytics_data.copy()
        if time_min:
            data = data[data["date_of_visit"] >= time_min]
        if time_max:
            data = data[data["date_of_visit"] <= time_max]
        if traffic_source:
            data["visits_from_source"] = (data["traffic_source"] == traffic_source).astype(int)
            return (
                data.groupby("date_of_visit").sum(numeric_only=True)["visits_from_source"].to_dict()
            )
        return data.groupby("date_of_visit").size().to_dict()

    def _tool_analytics__get_average_session_duration(
        self, time_min: str | None = None, time_max: str | None = None
    ) -> dict[str, float]:
        data = self.analytics_data.copy()
        if time_min:
            data = data[data["date_of_visit"] >= time_min]
        if time_max:
            data = data[data["date_of_visit"] <= time_max]
        data["session_duration_seconds"] = data["session_duration_seconds"].astype(float)
        return (
            data[["date_of_visit", "session_duration_seconds"]]
            .groupby("date_of_visit")
            .mean(numeric_only=True)["session_duration_seconds"]
            .to_dict()
        )

    def _tool_project_management__get_task_information_by_id(
        self, task_id: str | None = None, field: str | None = None
    ) -> Any:
        if not task_id:
            return "Task ID not provided."
        if not field:
            return "Field not provided."
        task = self.project_tasks[self.project_tasks["task_id"] == str(task_id)].to_dict(
            orient="records"
        )
        if not task:
            return "Task not found."
        return {field: task[0][field]} if field in task[0] else "Field not found."

    def _tool_project_management__search_tasks(
        self,
        task_name: str | None = None,
        assigned_to_email: str | None = None,
        list_name: str | None = None,
        due_date: str | None = None,
        board: str | None = None,
    ) -> Any:
        if not any([task_name, assigned_to_email, list_name, due_date, board]):
            return "No search parameters provided."
        tasks = self.project_tasks.copy()
        if task_name:
            tasks = tasks[tasks["task_name"].str.contains(task_name, case=False, na=False)]
        if assigned_to_email:
            tasks = tasks[
                tasks["assigned_to_email"].str.contains(assigned_to_email, case=False, na=False)
            ]
        if list_name:
            tasks = tasks[tasks["list_name"].str.contains(list_name, case=False, na=False)]
        if due_date:
            tasks = tasks[tasks["due_date"].str.contains(due_date, case=False, na=False)]
        if board:
            tasks = tasks[tasks["board"].str.contains(board, case=False, na=False)]
        return tasks.to_dict(orient="records")

    def _tool_project_management__create_task(
        self,
        task_name: str | None = None,
        assigned_to_email: str | None = None,
        list_name: str | None = None,
        due_date: str | None = None,
        board: str | None = None,
    ) -> str:
        if not all([task_name, assigned_to_email, list_name, due_date, board]):
            return "Missing task details."
        assigned_to_email = str(assigned_to_email).lower()
        if assigned_to_email not in self.project_tasks["assigned_to_email"].str.lower().values:
            return "Assignee email not valid. Please choose from the list of team members."
        if list_name not in ["Backlog", "In Progress", "In Review", "Completed"]:
            return "List not valid. Please choose from: 'Backlog', 'In Progress', 'In Review', 'Completed'."
        if board not in ["Back end", "Front end", "Design"]:
            return "Board not valid. Please choose from: 'Back end', 'Front end', 'Design'."
        task_id = str(int(self.project_tasks["task_id"].max()) + 1).zfill(8)
        new_task = pd.DataFrame(
            {
                "task_id": [task_id],
                "task_name": [task_name],
                "assigned_to_email": [assigned_to_email],
                "list_name": [list_name],
                "due_date": [due_date],
                "board": [board],
            }
        )
        self.project_tasks = pd.concat([self.project_tasks, new_task], ignore_index=True)
        return task_id

    def _tool_project_management__delete_task(self, task_id: str | None = None) -> str:
        if not task_id:
            return "Task ID not provided."
        if task_id in self.project_tasks["task_id"].values:
            self.project_tasks = self.project_tasks[self.project_tasks["task_id"] != task_id]
            return "Task deleted successfully."
        return "Task not found."

    def _tool_project_management__update_task(
        self, task_id: str | None = None, field: str | None = None, new_value: str | None = None
    ) -> str:
        if not task_id or not field or new_value is None:
            return "Task ID, field, or new value not provided."
        if field == "assigned_to_email":
            new_value = str(new_value).lower()
        if field == "board" and new_value not in ["Back end", "Front end", "Design"]:
            return "Board not valid. Please choose from: 'Back end', 'Front end', 'Design'."
        if field == "list_name" and new_value not in [
            "Backlog",
            "In Progress",
            "In Review",
            "Completed",
        ]:
            return "List not valid. Please choose from: 'Backlog', 'In Progress', 'In Review', 'Completed'."
        if (
            field == "assigned_to_email"
            and str(new_value) not in self.project_tasks["assigned_to_email"].str.lower().values
        ):
            return "Assignee email not valid. Please choose from the list of team members."
        if task_id in self.project_tasks["task_id"].values:
            if field in self.project_tasks.columns:
                self.project_tasks.loc[self.project_tasks["task_id"] == task_id, field] = new_value
                return "Task updated successfully."
            return "Field not valid."
        return "Task not found."

    def _tool_customer_relationship_manager__search_customers(
        self,
        customer_name: str | None = None,
        customer_email: str | None = None,
        product_interest: str | None = None,
        status: str | None = None,
        assigned_to_email: str | None = None,
        last_contact_date_min: str | None = None,
        last_contact_date_max: str | None = None,
        follow_up_by_min: str | None = None,
        follow_up_by_max: str | None = None,
    ) -> Any:
        if not any(
            [
                customer_name,
                customer_email,
                product_interest,
                status,
                assigned_to_email,
                last_contact_date_min,
                last_contact_date_max,
                follow_up_by_min,
                follow_up_by_max,
            ]
        ):
            return "No search parameters provided. Please provide at least one parameter."
        customers = self.crm_data.copy()
        if customer_name:
            customers = customers[
                customers["customer_name"].str.contains(customer_name, case=False, na=False)
            ]
        if customer_email:
            customers = customers[
                customers["customer_email"].str.contains(customer_email, case=False, na=False)
            ]
        if product_interest:
            customers = customers[
                customers["product_interest"].str.contains(product_interest, case=False, na=False)
            ]
        if status:
            customers = customers[customers["status"].str.contains(status, case=False, na=False)]
        if assigned_to_email:
            customers = customers[
                customers["assigned_to_email"].str.contains(assigned_to_email, case=False, na=False)
            ]
        if last_contact_date_min:
            customers = customers[customers["last_contact_date"] >= last_contact_date_min]
        if last_contact_date_max:
            customers = customers[customers["last_contact_date"] <= last_contact_date_max]
        if follow_up_by_min:
            customers = customers[customers["follow_up_by"] >= follow_up_by_min]
        if follow_up_by_max:
            customers = customers[customers["follow_up_by"] <= follow_up_by_max]
        return customers.to_dict(orient="records")[:5]

    def _tool_customer_relationship_manager__update_customer(
        self, customer_id: str | None = None, field: str | None = None, new_value: str | None = None
    ) -> str:
        if not customer_id or not field or new_value is None:
            return "Customer ID, field, or new value not provided."
        if field == "status" and new_value not in ["Qualified", "Won", "Lost", "Lead", "Proposal"]:
            return "Status not valid. Please choose from: 'Qualified', 'Won', 'Lost', 'Lead', 'Proposal'"
        if field == "product_interest" and new_value not in [
            "Software",
            "Hardware",
            "Services",
            "Consulting",
            "Training",
        ]:
            return (
                "Product interest not valid. Please choose from: 'Software', 'Hardware', "
                "'Services', 'Consulting', 'Training'"
            )
        if field in {"customer_email", "assigned_to_email"}:
            new_value = str(new_value).lower()
        if customer_id in self.crm_data["customer_id"].values:
            if field in self.crm_data.columns:
                self.crm_data.loc[self.crm_data["customer_id"] == customer_id, field] = new_value
                return "Customer updated successfully."
            return (
                "Field not valid. Please choose from: 'customer_name', 'assigned_to_email', "
                "'customer_email', 'customer_phone', 'last_contact_date', 'product_interest', "
                "'status', 'notes', 'follow_up_by'"
            )
        return "Customer not found."

    def _tool_customer_relationship_manager__add_customer(
        self,
        customer_name: str | None = None,
        assigned_to_email: str | None = None,
        status: str | None = None,
        customer_email: str | None = None,
        customer_phone: str | None = None,
        last_contact_date: str | None = None,
        product_interest: str | None = None,
        notes: str = "",
        follow_up_by: str | None = None,
    ) -> str:
        if not all([customer_name, assigned_to_email, status]):
            return "Please provide all required fields: customer_name, assigned_to_email, status."
        assigned_to_email = str(assigned_to_email).lower()
        if customer_email:
            customer_email = str(customer_email).lower()
        new_id = str(int(self.crm_data["customer_id"].max()) + 1).zfill(8)
        new_customer = pd.DataFrame(
            {
                "customer_id": [new_id],
                "customer_name": [customer_name],
                "customer_email": [customer_email],
                "customer_phone": [customer_phone],
                "last_contact_date": [last_contact_date],
                "product_interest": [product_interest],
                "status": [status],
                "assigned_to_email": [assigned_to_email],
                "notes": [notes],
                "follow_up_by": [follow_up_by],
            }
        )
        self.crm_data = pd.concat([self.crm_data, new_customer], ignore_index=True)
        return new_id

    def _tool_customer_relationship_manager__delete_customer(
        self, customer_id: str | None = None
    ) -> str:
        if not customer_id:
            return "Customer ID not provided."
        if customer_id not in self.crm_data["customer_id"].values:
            return "Customer not found."
        self.crm_data = self.crm_data[self.crm_data["customer_id"] != customer_id]
        return "Customer deleted successfully."

    def _tool_company_directory__find_email_address(self, name: str = "") -> Any:
        if name == "":
            return "Name not provided."
        matches = self.company_directory[
            self.company_directory["email_address"].str.contains(str(name).lower(), na=False)
        ]
        return matches["email_address"].values


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "calendar.get_event_information_by_id",
        "domain": "calendar",
        "description": "Returns the event for a given ID.",
        "properties": {
            "event_id": {"type": "string", "description": "8-digit ID of the event."},
            "field": {
                "type": "string",
                "description": 'Field to return. Available: "event_id", "event_name", "participant_email", "event_start", "duration".',
            },
        },
    },
    {
        "name": "calendar.search_events",
        "domain": "calendar",
        "description": "Returns events matching the query and optional time bounds.",
        "properties": {
            "query": {"type": "string", "description": "Query to search for."},
            "time_min": {
                "type": "string",
                "description": 'Lower bound format "YYYY-MM-DD HH:MM:SS".',
            },
            "time_max": {
                "type": "string",
                "description": 'Upper bound format "YYYY-MM-DD HH:MM:SS".',
            },
        },
    },
    {
        "name": "calendar.create_event",
        "domain": "calendar",
        "description": "Creates a new event.",
        "required": ["event_name", "participant_email", "event_start", "duration"],
        "properties": {
            "event_name": {"type": "string"},
            "participant_email": {"type": "string"},
            "event_start": {"type": "string"},
            "duration": {
                "type": "string",
                "description": 'Duration in whole minutes as a numeric string, e.g. "30", "60", or "120".',
            },
        },
    },
    {
        "name": "calendar.delete_event",
        "domain": "calendar",
        "description": "Deletes an event by ID.",
        "properties": {"event_id": {"type": "string"}},
    },
    {
        "name": "calendar.update_event",
        "domain": "calendar",
        "description": "Updates an event field by ID.",
        "properties": {
            "event_id": {"type": "string"},
            "field": {"type": "string"},
            "new_value": {
                "type": "string",
                "description": 'New field value. If updating duration, use whole minutes as a numeric string such as "30", "60", or "120".',
            },
        },
    },
    {
        "name": "email.get_email_information_by_id",
        "domain": "email",
        "description": "Retrieves specific details of an email by ID.",
        "properties": {"email_id": {"type": "string"}, "field": {"type": "string"}},
    },
    {
        "name": "email.search_emails",
        "domain": "email",
        "description": "Searches emails by query and optional date bounds.",
        "properties": {
            "query": {"type": "string"},
            "date_min": {"type": "string"},
            "date_max": {"type": "string"},
        },
    },
    {
        "name": "email.send_email",
        "domain": "email",
        "description": "Sends an email.",
        "properties": {
            "recipient": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
        },
    },
    {
        "name": "email.delete_email",
        "domain": "email",
        "description": "Deletes an email by ID.",
        "properties": {"email_id": {"type": "string"}},
    },
    {
        "name": "email.forward_email",
        "domain": "email",
        "description": "Forwards an email to a recipient.",
        "properties": {"email_id": {"type": "string"}, "recipient": {"type": "string"}},
    },
    {
        "name": "email.reply_email",
        "domain": "email",
        "description": "Replies to an email by ID.",
        "properties": {"email_id": {"type": "string"}, "body": {"type": "string"}},
    },
    {
        "name": "analytics.get_visitor_information_by_id",
        "domain": "analytics",
        "description": "Returns analytics data for a given visitor ID.",
        "properties": {"visitor_id": {"type": "string"}},
    },
    {
        "name": "analytics.create_plot",
        "domain": "analytics",
        "description": "Creates a plot file path for analytics data.",
        "properties": {
            "time_min": {"type": "string"},
            "time_max": {"type": "string"},
            "value_to_plot": {"type": "string"},
            "plot_type": {"type": "string"},
        },
    },
    {
        "name": "analytics.total_visits_count",
        "domain": "analytics",
        "description": "Returns total number of visits in a date range.",
        "properties": {"time_min": {"type": "string"}, "time_max": {"type": "string"}},
    },
    {
        "name": "analytics.engaged_users_count",
        "domain": "analytics",
        "description": "Returns engaged user counts in a date range.",
        "properties": {"time_min": {"type": "string"}, "time_max": {"type": "string"}},
    },
    {
        "name": "analytics.traffic_source_count",
        "domain": "analytics",
        "description": "Returns counts for a traffic source in a date range.",
        "properties": {
            "time_min": {"type": "string"},
            "time_max": {"type": "string"},
            "traffic_source": {"type": "string"},
        },
    },
    {
        "name": "analytics.get_average_session_duration",
        "domain": "analytics",
        "description": "Returns average session duration in a date range.",
        "properties": {"time_min": {"type": "string"}, "time_max": {"type": "string"}},
    },
    {
        "name": "project_management.get_task_information_by_id",
        "domain": "project_management",
        "description": "Returns task information for a given ID.",
        "properties": {"task_id": {"type": "string"}, "field": {"type": "string"}},
    },
    {
        "name": "project_management.search_tasks",
        "domain": "project_management",
        "description": "Searches tasks by task metadata.",
        "properties": {
            "task_name": {"type": "string"},
            "assigned_to_email": {"type": "string"},
            "list_name": {"type": "string"},
            "due_date": {"type": "string"},
            "board": {"type": "string"},
        },
    },
    {
        "name": "project_management.create_task",
        "domain": "project_management",
        "description": "Creates a new project task.",
        "properties": {
            "task_name": {"type": "string"},
            "assigned_to_email": {"type": "string"},
            "list_name": {"type": "string"},
            "due_date": {"type": "string"},
            "board": {"type": "string"},
        },
    },
    {
        "name": "project_management.delete_task",
        "domain": "project_management",
        "description": "Deletes a task by ID.",
        "properties": {"task_id": {"type": "string"}},
    },
    {
        "name": "project_management.update_task",
        "domain": "project_management",
        "description": "Updates a task field by ID.",
        "properties": {
            "task_id": {"type": "string"},
            "field": {"type": "string"},
            "new_value": {"type": "string"},
        },
    },
    {
        "name": "customer_relationship_manager.search_customers",
        "domain": "customer_relationship_manager",
        "description": "Searches customer records by multiple filters.",
        "properties": {
            "customer_name": {"type": "string"},
            "customer_email": {"type": "string"},
            "product_interest": {"type": "string"},
            "status": {"type": "string"},
            "assigned_to_email": {"type": "string"},
            "last_contact_date_min": {"type": "string"},
            "last_contact_date_max": {"type": "string"},
            "follow_up_by_min": {"type": "string"},
            "follow_up_by_max": {"type": "string"},
        },
    },
    {
        "name": "customer_relationship_manager.update_customer",
        "domain": "customer_relationship_manager",
        "description": "Updates a customer record by ID.",
        "properties": {
            "customer_id": {"type": "string"},
            "field": {"type": "string"},
            "new_value": {"type": "string"},
        },
    },
    {
        "name": "customer_relationship_manager.add_customer",
        "domain": "customer_relationship_manager",
        "description": "Adds a new customer record.",
        "properties": {
            "customer_name": {"type": "string"},
            "assigned_to_email": {"type": "string"},
            "status": {"type": "string"},
            "customer_email": {"type": "string"},
            "customer_phone": {"type": "string"},
            "last_contact_date": {"type": "string"},
            "product_interest": {"type": "string"},
            "notes": {"type": "string"},
            "follow_up_by": {"type": "string"},
        },
    },
    {
        "name": "customer_relationship_manager.delete_customer",
        "domain": "customer_relationship_manager",
        "description": "Deletes a customer record by ID.",
        "properties": {"customer_id": {"type": "string"}},
    },
    {
        "name": "company_directory.find_email_address",
        "domain": "company_directory",
        "description": "Finds the email address of an employee by name.",
        "properties": {"name": {"type": "string"}},
    },
]


def _parse_action_call(action: str) -> tuple[str, dict[str, Any]] | None:
    try:
        expr = ast.parse(action, mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(expr, ast.Call):
        return None
    func = expr.func
    if not (isinstance(func, ast.Attribute) and func.attr == "func"):
        return None
    if not (isinstance(func.value, ast.Attribute) and isinstance(func.value.value, ast.Name)):
        return None
    tool_name = f"{func.value.value.id}.{func.value.attr}"
    kwargs: dict[str, Any] = {}
    for kw in expr.keywords:
        if kw.arg is None:
            continue
        try:
            kwargs[kw.arg] = ast.literal_eval(kw.value)
        except Exception:
            kwargs[kw.arg] = None
    return tool_name, kwargs


def _format_function_call(tool_name: str, arguments: dict[str, Any]) -> str:
    parts = []
    for key, value in arguments.items():
        escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
        parts.append(f'{key}="{escaped}"')
    return f"{tool_name}.func({', '.join(parts)})"


def _normalize_state(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col in FIELDS_NOT_TO_LOWER:
            continue
        if pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].astype(str).str.lower()
        else:
            out[col] = out[col].astype(str).str.lower()
    return out.reset_index(drop=True)


def _normalize_duration_value(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return str(value)

    if re.fullmatch(r"\d+", text):
        return text

    clock_match = re.fullmatch(r"(\d+):(\d{1,2})", text)
    if clock_match:
        hours = int(clock_match.group(1))
        minutes = int(clock_match.group(2))
        return str(hours * 60 + minutes)

    minute_match = re.fullmatch(r"(\d+)\s*(minute|minutes|min|mins)", text)
    if minute_match:
        return str(int(minute_match.group(1)))

    hour_match = re.fullmatch(r"(\d+)\s*(hour|hours|hr|hrs)", text)
    if hour_match:
        return str(int(hour_match.group(1)) * 60)

    return str(value)


class WorkBenchBenchmark:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.domain: str = str(cfg.get("domain", "multi_domain"))
        self.tool_selection: str = str(cfg.get("tool_selection", "domains"))
        self.max_tool_iterations: int = int(cfg.get("max_tool_iterations", 20))
        self.tasks_version: str = str(cfg.get("tasks_version", "v1")).strip()
        self.data_dir = Path(str(cfg.get("data_dir", ".cache/workbench"))).expanduser()
        self._ensure_data_exists()

    def _query_files(self) -> dict[str, str]:
        version_part = "" if self.tasks_version in {"", "current", "v2"} else f"{self.tasks_version}/"
        return {
            domain: f"data/processed/tasks_and_outcomes/{version_part}{filename}"
            for domain, filename in QUERY_FILE_NAMES.items()
        }

    def _ensure_data_exists(self) -> None:
        for rel_path in [*DATA_FILES, *self._query_files().values()]:
            target = self.data_dir / rel_path
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            url = RAW_BASE_URL + rel_path
            logger.info(f"Downloading WorkBench asset: {url}")
            with urllib.request.urlopen(url, timeout=30) as response:
                target.write_bytes(response.read())

    def load_tasks(self, task_limit: int | None = None) -> Sequence[BenchmarkTask]:
        query_files = self._query_files()
        if self.domain not in query_files:
            raise ValueError(f"Unknown WorkBench domain '{self.domain}'")
        query_path = self.data_dir / query_files[self.domain]
        tasks: list[BenchmarkTask] = []
        with query_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader):
                query = str(row.get("query") or row.get("task") or "")
                answer_raw = row.get("answer") or row.get("outcome") or "[]"
                answer = ast.literal_eval(answer_raw)
                domains = ast.literal_eval(row["domains"])
                tasks.append(
                    BenchmarkTask(
                        task_id=f"{self.domain}_{idx}",
                        prompt=query,
                        reference_answer="",
                        metadata={
                            "query": query,
                            "answer": answer,
                            "domains": domains,
                            "base_template": row.get("base_template", ""),
                            "chosen_template": row.get("chosen_template", ""),
                        },
                    )
                )
                if task_limit is not None and len(tasks) >= task_limit:
                    break
        return tasks

    def _build_tools(
        self, sandbox: WorkBenchSandbox, task_domains: list[str]
    ) -> list[dict[str, Any]]:
        resolved_domains = [DOMAIN_ALIASES.get(domain, domain) for domain in task_domains]
        active_domains = set(CORE_DOMAINS if self.tool_selection == "all" else resolved_domains)
        active_domains.add("company_directory")
        tools: list[dict[str, Any]] = []
        for spec in TOOL_SPECS:
            if spec["domain"] not in active_domains:
                continue
            name = str(spec["name"])
            tools.append(
                {
                    "name": name,
                    "description": str(spec["description"]),
                    "parameters": {
                        "type": "object",
                        "properties": dict(spec["properties"]),
                        "required": list(spec.get("required", [])),
                    },
                    "handler": lambda args, tool_name=name, sb=sandbox: sb.invoke(tool_name, args),
                    "side_effect": name in SIDE_EFFECT_TOOLS,
                    # Runs the call on a throwaway copy; systems that defer writes use it to
                    # validate a proposed write without changing the shared state.
                    "dry_run_handler": (
                        lambda args, tool_name=name, sb=sandbox: sb.fork().invoke(tool_name, args)
                    ),
                }
            )
        return tools

    def run(
        self,
        task: BenchmarkTask,
        runner: MASRunner,
        run_index: int,
        seed: int,
    ) -> MASRunResult:
        sandbox = WorkBenchSandbox(self.data_dir)
        tools = self._build_tools(sandbox, list(task.metadata.get("domains", [])))
        prompt = [
            {
                "role": "system",
                "content": (
                    f"Today's date is {HARDCODED_CURRENT_TIME.strftime('%A')}, "
                    f"{HARDCODED_CURRENT_TIME.date()} and the current time is {HARDCODED_CURRENT_TIME.time()}. "
                    "Remember the current date and time when answering queries. "
                    "Meetings must not start before 9am or end after 6pm. "
                    "Use the provided workplace tools to complete the task. "
                    "After using tools, provide a brief natural-language confirmation."
                ),
            },
            {"role": "user", "content": task.metadata["query"]},
        ]
        step_task = BenchmarkTask(
            task_id=task.task_id,
            prompt=prompt,
            reference_answer="",
            metadata=dict(task.metadata),
        )
        result = runner.run_task(
            task=step_task,
            run_index=run_index,
            seed=seed,
            tools=tools,
            max_tool_iterations=self.max_tool_iterations,
            benchmark_name="workbench",
        )
        function_calls = self._extract_selected_function_calls(
            dict(result.run_metadata),
            result.trace_events,
        )
        return MASRunResult(
            final_answer=result.final_answer,
            trace_events=result.trace_events,
            run_metadata={
                **dict(result.run_metadata),
                "function_calls": function_calls,
                "error": "",
            },
        )

    @staticmethod
    def _extract_function_calls(trace_events: list[Any]) -> list[str]:
        calls: list[str] = []
        for event in trace_events:
            event_type = getattr(event, "event_type", None)
            payload = getattr(event, "payload", None)
            if event_type != "tool_call" or not isinstance(payload, dict):
                continue
            tool_name = str(payload.get("tool_name", "")).strip()
            if not tool_name or tool_name == "inter_agent_send":
                continue
            arguments = payload.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            calls.append(_format_function_call(tool_name, arguments))
        return calls

    @classmethod
    def _extract_function_calls_from_tool_records(cls, tool_records: list[Any]) -> list[str]:
        calls: list[str] = []
        for record in tool_records:
            if not isinstance(record, dict):
                continue
            tool_name = str(record.get("tool_name", "")).strip()
            if not tool_name or tool_name == "inter_agent_send":
                continue
            arguments = record.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            calls.append(_format_function_call(tool_name, arguments))
        return calls

    @classmethod
    def _extract_selected_function_calls(
        cls,
        run_metadata: dict[str, Any],
        trace_events: list[Any],
    ) -> list[str]:
        artifact_records = run_metadata.get("artifact_records", [])
        selected_artifact_id = run_metadata.get("selected_artifact_id")
        committed = run_metadata.get("committed_actions")
        if isinstance(committed, list):
            # Commit-once runs: writes made during the run were only proposals; the
            # committed list is the only set of writes that reached the environment.
            lineage = (
                collect_lineage_tool_records(artifact_records, selected_artifact_id)
                if isinstance(artifact_records, list) and selected_artifact_id
                else []
            )
            reads = [
                record
                for record in lineage
                if isinstance(record, dict)
                and str(record.get("tool_name", "")) not in SIDE_EFFECT_TOOLS
            ]
            writes = [
                record
                for record in committed
                if isinstance(record, dict) and str(record.get("status", "")) == "completed"
            ]
            return cls._extract_function_calls_from_tool_records(reads + writes)
        if isinstance(artifact_records, list) and selected_artifact_id:
            lineage_tool_records = collect_lineage_tool_records(artifact_records, selected_artifact_id)
            if lineage_tool_records:
                return cls._extract_function_calls_from_tool_records(lineage_tool_records)
        return cls._extract_function_calls(trace_events)

    def _execute_actions_and_reset_state(
        self, actions: list[str]
    ) -> tuple[bool, dict[str, pd.DataFrame]]:
        sandbox = WorkBenchSandbox(self.data_dir)
        for action in actions:
            parsed = _parse_action_call(action)
            if parsed is None:
                continue
            tool_name, kwargs = parsed
            try:
                sandbox.invoke(tool_name, kwargs)
            except Exception:
                continue
        return True, sandbox.snapshot()

    @staticmethod
    def _get_function_name(action: str) -> str:
        return ".".join(action.split("(")[0].split(".")[0:2])

    def _is_exact_match(
        self, predicted_actions: list[str], ground_truth_actions: list[str]
    ) -> bool:
        predicted = [
            action
            for action in predicted_actions
            if self._get_function_name(action) in SIDE_EFFECT_TOOLS
        ]
        predicted = sorted(action.lower() for action in predicted)
        ground_truth = sorted(action.lower() for action in ground_truth_actions)
        return predicted == ground_truth

    def _is_correct(
        self, predicted_actions: list[str], ground_truth_actions: list[str], error: str
    ) -> bool:
        if error:
            return False
        _, predicted_state = self._execute_actions_and_reset_state(predicted_actions)
        _, ground_truth_state = self._execute_actions_and_reset_state(ground_truth_actions)
        return all(
            _normalize_state(predicted_state[name]).equals(
                _normalize_state(ground_truth_state[name])
            )
            for name in predicted_state
        )

    def _has_side_effects(
        self, predicted_actions: list[str], ground_truth_actions: list[str]
    ) -> bool:
        original_state = WorkBenchSandbox(self.data_dir).snapshot()
        _, predicted_state = self._execute_actions_and_reset_state(predicted_actions)
        state_changed = any(
            not _normalize_state(predicted_state[name]).equals(
                _normalize_state(original_state[name])
            )
            for name in predicted_state
        )
        return state_changed and not self._is_correct(predicted_actions, ground_truth_actions, "")

    def evaluate(
        self,
        task: BenchmarkTask,
        prediction: str,
        *,
        run_metadata: dict[str, Any] | None = None,
    ) -> BenchmarkEvaluation:
        run_metadata = run_metadata or {}
        ground_truth_actions = list(task.metadata.get("answer", []))
        predicted_actions = list(run_metadata.get("function_calls", []))
        error = str(run_metadata.get("error", ""))
        prediction_text = str(prediction or "")
        substantive_prediction = has_substantive_answer(prediction_text)

        exact_match = self._is_exact_match(predicted_actions, ground_truth_actions)
        correct = self._is_correct(predicted_actions, ground_truth_actions, error)
        unwanted_side_effects = self._has_side_effects(predicted_actions, ground_truth_actions)
        if not ground_truth_actions:
            exact_match = exact_match and substantive_prediction
            correct = correct and substantive_prediction

        return BenchmarkEvaluation(
            task_id=task.task_id,
            score=1.0 if correct else 0.0,
            success=correct,
            details={
                "query": task.metadata.get("query", ""),
                "ground_truth": ground_truth_actions,
                "prediction": predicted_actions,
                "error": error,
                "exact_match": exact_match,
                "correct": correct,
                "prediction_substantive": substantive_prediction,
                "unwanted_side_effects": unwanted_side_effects,
                "tool_selection": self.tool_selection,
                "domains": task.metadata.get("domains", []),
            },
        )

    def requirements(self) -> dict[str, Any]:
        return {
            "benchmark": "workbench",
            "source_repo": "https://github.com/olly-styles/WorkBench",
            "dataset": f"WorkBench queries for domain={self.domain}",
            "notes": [
                "Uses official WorkBench processed CSVs and query/answer pairs.",
                "Company directory tool is always included, matching upstream behavior.",
                "Evaluation follows official state-change correctness semantics.",
            ],
        }
