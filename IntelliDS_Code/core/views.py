from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from django.contrib.auth.models import User
from django.core.paginator import Paginator
from django.http import JsonResponse
from functools import wraps
from .models import Menu, UserAccess
from django.urls import resolve
from django.views.decorators.csrf import csrf_protect
import json
from django.template.loader import render_to_string
from .automl.forms import DatasetUploadForm
from .automl.utils import read_dataset
from io import StringIO
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
import uuid
import pandas as pd
import numpy as np
import os
import json
import pickle
import seaborn as sns
import re
from django.shortcuts import render
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.urls import reverse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
)

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)

from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
)

from sklearn.svm import (
    SVC,
    SVR,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import uuid
import plotly.express as px
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.shortcuts import render


def is_admin(user):
    """Check if user is admin (superuser)."""
    return user.is_superuser


def menu_access_required(menu_url_name):
    """
    Decorator to check if the user has access to a specific menu by url_name.
    - Admins have unrestricted access.
    - Regular users must have that menu assigned in UserAccess.
    """

    def decorator(view_func):
        @wraps(view_func)
        @login_required
        def _wrapped_view(request, *args, **kwargs):
            if request.user.is_superuser:
                return view_func(request, *args, **kwargs)

            try:
                menu = Menu.objects.get(url_name=menu_url_name)
                if UserAccess.objects.filter(user=request.user, menu=menu).exists():
                    return view_func(request, *args, **kwargs)
                else:
                    messages.error(
                        request,
                        f"You do not have access to '{menu.name}'. Please contact admin.",
                    )
            except Menu.DoesNotExist:
                messages.error(request, "Requested page does not exist.")

            return redirect("user_dashboard")

        return _wrapped_view

    return decorator


@csrf_protect
def logout_and_redirect(request):
    """Logout the user and redirect to login page via POST."""
    if request.method == "POST":
        logout(request)
        return redirect("login")
    # Optional: if someone tries GET, just redirect to login
    return redirect("login")


# === Error handlers ===
def handler_400(request, exception):
    return render(request, "errors/400.html", status=400)


def handler401(request, exception=None):
    return render(request, "errors/401.html", status=401)


def handler403(request, exception):
    return render(request, "errors/403.html", status=403)


def handler404(request, exception):
    return render(request, "errors/404.html", status=404)


def handler500(request):
    return render(request, "errors/500.html", status=500)


def handler502(request, exception=None):
    return render(request, "errors/502.html", status=502)


def handler503(request, exception=None):
    return render(request, "errors/503.html", status=503)


def login_view(request):
    """Login for admin and users."""
    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "").strip()

        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            messages.success(request, f"Welcome back, {user.username}!")
            return redirect(
                "admin_dashboard" if user.is_superuser else "user_dashboard"
            )
        else:
            messages.error(request, "Invalid username or password.")

    return render(request, "core/login.html")


@login_required
def logout_view(request):
    """Logout the current user."""
    logout(request)
    messages.success(request, "You have been logged out.")
    return redirect("login")


@login_required
@user_passes_test(is_admin)
def admin_dashboard(request):
    """Admin dashboard: manage users, menus, and user access."""

    # --- Users with optional filters ---
    query = request.GET.get("query", "")
    status_filter = request.GET.get("status", "")
    users_list = User.objects.filter(username__icontains=query).order_by("id")

    if status_filter == "active":
        users_list = users_list.filter(is_active=True)
    elif status_filter == "inactive":
        users_list = users_list.filter(is_active=False)

    paginator = Paginator(users_list, 5)
    page_number = request.GET.get("page")
    users = paginator.get_page(page_number)

    # --- Menus (all for admin) ---
    menus_list = Menu.objects.all().order_by("order")
    menu_paginator = Paginator(menus_list, 5)  # 5 menus per page
    menu_page_number = request.GET.get("menu_page")
    menus = menu_paginator.get_page(menu_page_number)

    # Current URL name for sidebar highlighting
    current_url_name = resolve(request.path_info).url_name

    # ------------------- Handle POST Actions -------------------
    if request.method == "POST":
        action = request.POST.get("action")

        # ---- User Management ----
        if action == "add":
            username = request.POST.get("username").strip()
            email = request.POST.get("email").strip()
            password = request.POST.get("password").strip()
            is_active = request.POST.get("is_active") == "Active"
            is_superuser = request.POST.get("is_superuser") == "Admin"

            if User.objects.filter(username=username).exists():
                messages.error(request, "Username already exists!")
            else:
                User.objects.create_user(
                    username=username,
                    email=email,
                    password=password,
                    is_active=is_active,
                    is_superuser=is_superuser,
                )
                messages.success(request, "User created successfully!")
            return redirect("admin_dashboard")

        elif action == "edit":
            user_id = request.POST.get("user_id")
            user_obj = get_object_or_404(User, id=user_id)
            username = request.POST.get("username").strip()
            email = request.POST.get("email").strip()
            password = request.POST.get("password").strip()
            is_active = request.POST.get("is_active") == "Active"
            is_superuser = request.POST.get("is_superuser") == "Admin"

            if (
                username != user_obj.username
                and User.objects.filter(username=username).exists()
            ):
                messages.error(request, "Username already exists!")
            elif user_obj == request.user and not is_active:
                messages.error(request, "You cannot deactivate yourself!")
            else:
                user_obj.username = username
                user_obj.email = email
                user_obj.is_active = is_active
                user_obj.is_superuser = is_superuser
                if password:
                    user_obj.set_password(password)
                user_obj.save()
                messages.success(request, "User updated successfully!")
            return redirect("admin_dashboard")

        elif action == "delete":
            user_id = request.POST.get("user_id")
            user_obj = get_object_or_404(User, id=user_id)
            if user_obj == request.user:
                messages.error(request, "You cannot delete yourself!")
            else:
                user_obj.delete()
                messages.success(request, "User deleted successfully!")
            return redirect("admin_dashboard")

        # ---- Menu Management ----
        elif action == "add_menu":
            name = request.POST.get("name").strip()
            url_name = request.POST.get("url_name").strip()
            icon = request.POST.get("icon", "").strip()
            order = Menu.objects.count() + 1

            if Menu.objects.filter(name=name).exists():
                messages.error(request, "Menu with this name already exists!")
            else:
                Menu.objects.create(
                    name=name, url_name=url_name, icon=icon, order=order
                )
                messages.success(request, "Menu added successfully!")
            return redirect("admin_dashboard")

        elif action == "edit_menu":
            menu_id = request.POST.get("menu_id")
            menu_obj = get_object_or_404(Menu, id=menu_id)
            name = request.POST.get("name").strip()
            url_name = request.POST.get("url_name").strip()
            icon = request.POST.get("icon", "").strip()

            if name != menu_obj.name and Menu.objects.filter(name=name).exists():
                messages.error(request, "Another menu with this name exists!")
            else:
                menu_obj.name = name
                menu_obj.url_name = url_name
                menu_obj.icon = icon
                menu_obj.save()
                messages.success(request, "Menu updated successfully!")
            return redirect("admin_dashboard")

        elif action == "delete_menu":
            menu_id = request.POST.get("menu_id")
            menu_obj = get_object_or_404(Menu, id=menu_id)
            menu_obj.delete()
            messages.success(request, "Menu deleted successfully!")
            return redirect("admin_dashboard")

        # ---- User Access Management ----
        elif action == "access":
            user_id = request.POST.get("user")
            user_obj = get_object_or_404(User, id=user_id)
            menu_ids = request.POST.getlist("menus")

            # Remove old access and add new ones
            UserAccess.objects.filter(user=user_obj).delete()
            for mid in menu_ids:
                menu = get_object_or_404(Menu, id=mid)
                UserAccess.objects.create(user=user_obj, menu=menu)

            messages.success(request, f"Access updated for {user_obj.username}!")
            return redirect("admin_dashboard")
    total_menus = Menu.objects.count()
    context = {
        "users": users,
        "query": query,
        "status_filter": status_filter,
        "total_users": User.objects.count(),
        "total_admins": User.objects.filter(is_superuser=True).count(),
        "total_menus": total_menus,
        "menus": menus,
        "current_url_name": current_url_name,
    }
    return render(request, "core/admin_dashboard.html", context)


@login_required
def user_dashboard(request):
    """User dashboard: only show assigned menus."""
    menus = Menu.objects.filter(useraccess__user=request.user).order_by("order")
    current_url_name = resolve(request.path_info).url_name

    if request.method == "POST" and request.POST.get("action") == "edit_profile":
        email = request.POST.get("email").strip()
        password = request.POST.get("password").strip()
        user = request.user
        # username is NOT updated
        user.email = email
        if password:
            user.set_password(password)
        user.save()
        messages.success(request, "Profile updated successfully!")
        return redirect("user_dashboard")

    context = {"menus": menus, "current_url_name": current_url_name}
    return render(request, "core/user_dashboard.html", context)


@login_required
@user_passes_test(is_admin)
def manage_access(request, user_id=None):
    """Separate manage access page or modal."""
    menus = Menu.objects.all().order_by("order")

    if user_id:
        user_obj = get_object_or_404(User, id=user_id)
        current_access = UserAccess.objects.filter(user=user_obj).values_list(
            "menu_id", flat=True
        )

        if request.method == "POST":
            menu_ids = request.POST.getlist("menus")
            UserAccess.objects.filter(user=user_obj).delete()
            for mid in menu_ids:
                menu = get_object_or_404(Menu, id=mid)
                UserAccess.objects.create(user=user_obj, menu=menu)
            messages.success(request, f"Access updated for {user_obj.username}!")
            return redirect("manage_access")

        return render(
            request,
            "core/manage_access_modal.html",
            {"user_obj": user_obj, "menus": menus, "current_access": current_access},
        )

    users = User.objects.all().order_by("id")
    return render(
        request, "core/manage_access_modal.html", {"users": users, "menus": menus}
    )


@login_required
@user_passes_test(is_admin)
def get_user_menus(request, user_id):
    """AJAX endpoint to return a list of menu IDs assigned to a user."""
    user_obj = get_object_or_404(User, id=user_id)
    menu_ids = list(
        UserAccess.objects.filter(user=user_obj).values_list("menu_id", flat=True)
    )
    return JsonResponse({"menu_ids": menu_ids})


# =============================================================================================================


@login_required
def dynamic_menu_page(request, url_name):
    """
    Generic view to render pages dynamically based on the menu's URL name.
    Admin sees all menus, user sees assigned menus only.
    """
    try:
        menu = Menu.objects.get(url_name=url_name)
    except Menu.DoesNotExist:
        messages.error(request, "Requested page does not exist.")
        return redirect("user_dashboard")

    # Check access for non-admins
    if (
        not request.user.is_superuser
        and not UserAccess.objects.filter(user=request.user, menu=menu).exists()
    ):
        messages.error(request, f"You do not have access to '{menu.name}'.")
        return redirect("user_dashboard")

    if request.user.is_superuser:
        menus = Menu.objects.all().order_by("order")
    else:
        menus = Menu.objects.filter(useraccess__user=request.user).order_by("order")

    current_url_name = url_name
    template_name = f"core/{url_name}.html"

    return render(
        request,
        template_name,
        {"menus": menus, "current_url_name": current_url_name, "title": menu.name},
    )


# =======================================================================================================================


def upload_dataset(request):
    """
    Step 1
    Upload Dataset
    """

    form = DatasetUploadForm()

    preview = None
    summary = None
    columns = []
    profile = []
    quality = {}
    recommendations = []
    health_score = 100
    missing_columns = []
    identifier_columns = []
    ai_report = {}
    target_candidates = []
    numeric_statistics = []
    categorical_statistics = []
    column_categories = {
        "numeric": 0,
        "categorical": 0,
        "datetime": 0,
        "boolean": 0,
    }

    dataset_ready = False

    if request.method == "POST":

        form = DatasetUploadForm(request.POST, request.FILES)

        if form.is_valid():

            uploaded_file = form.cleaned_data["dataset"]

            try:

                dataframe = read_dataset(uploaded_file)

                dataframe.columns = [str(col).strip() for col in dataframe.columns]

                # =====================================================
                # Normalize Missing Values
                # =====================================================

                # Trim whitespace from string columns
                object_cols = dataframe.select_dtypes(include=["object"]).columns

                for col in object_cols:
                    dataframe[col] = dataframe[col].astype(str).str.strip()

                # Replace common missing value representations
                dataframe.replace(
                    [
                        "",
                        " ",
                        "NA",
                        "N/A",
                        "na",
                        "n/a",
                        "NULL",
                        "null",
                        "None",
                        "none",
                        "?",
                        "-",
                    ],
                    np.nan,
                    inplace=True,
                )

                # request.session["dataset"] = dataframe.to_json(orient="split")
                # request.session["dataset_name"] = uploaded_file.name

                # preview = dataframe.head(100)
                preview = dataframe.head(100).replace({np.nan: "NaN"}).values.tolist()
                columns = dataframe.columns.tolist()

                summary = {
                    "rows": dataframe.shape[0],
                    "columns": dataframe.shape[1],
                    "missing": int(dataframe.isna().sum().sum()),
                    "duplicates": int(dataframe.duplicated().sum()),
                    "memory": round(
                        dataframe.memory_usage(deep=True).sum() / 1024 / 1024,
                        2,
                    ),
                    "numeric": len(dataframe.select_dtypes(include="number").columns),
                    "categorical": len(
                        dataframe.select_dtypes(include="object").columns
                    ),
                    "datetime": len(
                        dataframe.select_dtypes(include=["datetime64"]).columns
                    ),
                    "boolean": len(dataframe.select_dtypes(include="bool").columns),
                }

                profile = []

                for column in dataframe.columns:

                    profile.append(
                        {
                            "name": column,
                            "dtype": str(dataframe[column].dtype),
                            "missing": int(dataframe[column].isna().sum()),
                            "missing_percent": round(
                                dataframe[column].isna().mean() * 100,
                                2,
                            ),
                            "unique": int(dataframe[column].nunique()),
                            "sample": (
                                ""
                                if dataframe[column].dropna().empty
                                else str(dataframe[column].dropna().iloc[0])
                            ),
                            "memory": round(
                                dataframe[column].memory_usage(deep=True) / 1024, 2
                            ),
                        }
                    )

                missing_columns = []

                for column in dataframe.columns:

                    missing = int(dataframe[column].isna().sum())

                    if missing > 0:

                        missing_columns.append(
                            {
                                "column": column,
                                "missing": missing,
                                "percent": round(
                                    dataframe[column].isna().mean() * 100, 2
                                ),
                            }
                        )
                quality = {
                    "empty_columns": int(
                        dataframe.columns[dataframe.isna().all()].shape[0]
                    ),
                    "constant_columns": int(
                        (dataframe.nunique(dropna=False) == 1).sum()
                    ),
                    "duplicate_rows": int(dataframe.duplicated().sum()),
                    "missing_cells": int(dataframe.isna().sum().sum()),
                }

                health_score = 100

                health_score -= min(quality["missing_cells"] * 0.05, 30)
                health_score -= min(quality["duplicate_rows"] * 2, 20)
                health_score -= quality["empty_columns"] * 10
                health_score -= quality["constant_columns"] * 5

                health_score = max(0, round(health_score))
                recommendations = []

                # =====================================================
                # Dataset Health
                # =====================================================

                if health_score >= 90:
                    recommendations.append(
                        {
                            "type": "success",
                            "title": "Excellent Dataset",
                            "message": "Dataset quality is excellent and is ready for machine learning.",
                        }
                    )

                elif health_score >= 75:
                    recommendations.append(
                        {
                            "type": "info",
                            "title": "Good Dataset",
                            "message": "Minor preprocessing is recommended before training.",
                        }
                    )

                else:
                    recommendations.append(
                        {
                            "type": "warning",
                            "title": "Dataset Needs Cleaning",
                            "message": "Several quality issues should be resolved before model training.",
                        }
                    )

                # =====================================================
                # Missing Values
                # =====================================================

                if quality["missing_cells"] > 0:

                    percent = round(
                        quality["missing_cells"]
                        / (summary["rows"] * summary["columns"])
                        * 100,
                        2,
                    )

                    recommendations.append(
                        {
                            "type": "warning",
                            "title": "Missing Values",
                            "message": f"{quality['missing_cells']} cells ({percent}%) are missing. "
                            "Consider mean/median imputation for numeric columns and mode imputation for categorical columns.",
                        }
                    )

                # =====================================================
                # Duplicate Rows
                # =====================================================

                if quality["duplicate_rows"] > 0:

                    recommendations.append(
                        {
                            "type": "warning",
                            "title": "Duplicate Records",
                            "message": f"{quality['duplicate_rows']} duplicate rows were detected. "
                            "Removing duplicates can improve model performance.",
                        }
                    )

                # =====================================================
                # Empty Columns
                # =====================================================

                if quality["empty_columns"] > 0:

                    recommendations.append(
                        {
                            "type": "danger",
                            "title": "Empty Columns",
                            "message": f"{quality['empty_columns']} completely empty columns should be removed.",
                        }
                    )

                # =====================================================
                # Constant Columns
                # =====================================================

                if quality["constant_columns"] > 0:

                    recommendations.append(
                        {
                            "type": "danger",
                            "title": "Constant Columns",
                            "message": f"{quality['constant_columns']} columns contain only one value. "
                            "They provide no predictive power.",
                        }
                    )

                # =====================================================
                # Identifier Columns
                # =====================================================

                identifier_columns = []

                for column in dataframe.columns:

                    if dataframe[column].nunique() == len(dataframe):

                        identifier_columns.append(column)

                if len(identifier_columns) > 0:

                    recommendations.append(
                        {
                            "type": "info",
                            "title": "Identifier Columns",
                            "message": f"{len(identifier_columns)} possible identifier column(s) detected. "
                            "These should usually be excluded from model training.",
                        }
                    )

                # =====================================================
                # Numeric / Categorical Balance
                # =====================================================

                if summary["numeric"] == 0:

                    recommendations.append(
                        {
                            "type": "warning",
                            "title": "No Numeric Features",
                            "message": "The dataset contains no numeric columns. Encoding may be required before model training.",
                        }
                    )

                if summary["categorical"] > summary["numeric"]:

                    recommendations.append(
                        {
                            "type": "info",
                            "title": "High Categorical Features",
                            "message": "Categorical features dominate this dataset. One-Hot or Label Encoding is recommended.",
                        }
                    )

                # =====================================================
                # Dataset Size
                # =====================================================

                if summary["rows"] < 100:

                    recommendations.append(
                        {
                            "type": "warning",
                            "title": "Small Dataset",
                            "message": "Very small datasets may lead to poor model generalization.",
                        }
                    )

                elif summary["rows"] > 100000:

                    recommendations.append(
                        {
                            "type": "info",
                            "title": "Large Dataset",
                            "message": "Large dataset detected. Training may take longer but should improve model accuracy.",
                        }
                    )

                # =====================================================
                # Target Suggestions
                # =====================================================

                if len(target_candidates) == 0:

                    recommendations.append(
                        {
                            "type": "warning",
                            "title": "Target Column",
                            "message": "No obvious target column was detected. Please select one manually.",
                        }
                    )

                else:

                    recommendations.append(
                        {
                            "type": "success",
                            "title": "Target Candidates",
                            "message": f"{len(target_candidates)} possible target column(s) were detected.",
                        }
                    )

                # =====================================================
                # Final Advice
                # =====================================================

                if dataset_ready:

                    recommendations.append(
                        {
                            "type": "success",
                            "title": "Ready for AutoML",
                            "message": "Dataset passed the initial quality assessment and can proceed to preprocessing.",
                        }
                    )

                # ======================================================
                # IntelliDS AI Target Detection Engine
                # ======================================================

                target_candidates = []

                target_keywords = [
                    "target",
                    "label",
                    "class",
                    "status",
                    "result",
                    "output",
                    "prediction",
                    "approved",
                    "churn",
                    "default",
                    "fraud",
                ]

                for column in dataframe.columns:

                    unique = dataframe[column].nunique()

                    if unique == len(dataframe):
                        continue

                    confidence = 0

                    # Column name intelligence
                    for word in target_keywords:

                        if word in column.lower():
                            confidence += 50

                    # Categorical target
                    if dataframe[column].dtype == "object":

                        if 2 <= unique <= 20:

                            confidence += 30

                    # Binary target
                    elif unique == 2:

                        confidence += 40

                    # Regression possibility
                    elif dataframe[column].dtype in ["int64", "float64"]:

                        if unique > 20:
                            confidence += 20

                    if confidence >= 30:

                        target_candidates.append(
                            {
                                "column": column,
                                "unique": unique,
                                "type": (
                                    "Regression" if unique > 20 else "Classification"
                                ),
                                "confidence": confidence,
                            }
                        )

                # Sort best target first

                target_candidates = sorted(
                    target_candidates, key=lambda x: x["confidence"], reverse=True
                )
                numeric_statistics = []

                numeric_df = dataframe.select_dtypes(include="number")

                for column in numeric_df.columns:

                    numeric_statistics.append(
                        {
                            "column": column,
                            "mean": round(numeric_df[column].mean(), 2),
                            "median": round(numeric_df[column].median(), 2),
                            "std": round(numeric_df[column].std(), 2),
                            "min": round(numeric_df[column].min(), 2),
                            "max": round(numeric_df[column].max(), 2),
                        }
                    )

                categorical_statistics = []

                cat_df = dataframe.select_dtypes(include="object")

                for column in cat_df.columns:

                    mode = cat_df[column].mode()

                    categorical_statistics.append(
                        {
                            "column": column,
                            "unique": int(cat_df[column].nunique()),
                            "top": "" if mode.empty else str(mode.iloc[0]),
                            "frequency": (
                                0
                                if mode.empty
                                else int((cat_df[column] == mode.iloc[0]).sum())
                            ),
                        }
                    )
                column_categories = {
                    "numeric": len(dataframe.select_dtypes(include="number").columns),
                    "categorical": len(
                        dataframe.select_dtypes(include="object").columns
                    ),
                    "datetime": len(
                        dataframe.select_dtypes(include=["datetime64"]).columns
                    ),
                    "boolean": len(dataframe.select_dtypes(include="bool").columns),
                }

                problem_type = "Unknown"

                if len(target_candidates):

                    target_unique = target_candidates[0]["unique"]

                    if target_unique <= 20:
                        problem_type = "Classification"
                    else:
                        problem_type = "Regression"

                # ======================================================
                # IntelliDS AI Model Recommendation Engine
                # ======================================================

                recommended_model = "Random Forest"

                numeric_features = summary["numeric"]
                categorical_features = summary["categorical"]
                rows = summary["rows"]
                columns_count = summary["columns"]

                # Classification Models
                if problem_type == "Classification":

                    if rows < 1000:

                        if categorical_features > numeric_features:
                            recommended_model = "Decision Tree Classifier"
                        else:
                            recommended_model = "Logistic Regression"

                    elif rows < 50000:

                        if columns_count < 20:
                            recommended_model = "Random Forest Classifier"
                        else:
                            recommended_model = "XGBoost Classifier"

                    else:

                        recommended_model = "LightGBM Classifier"

                # Regression Models
                elif problem_type == "Regression":

                    if rows < 1000:

                        recommended_model = "Linear Regression"

                    elif rows < 50000:

                        if columns_count < 20:
                            recommended_model = "Random Forest Regressor"
                        else:
                            recommended_model = "XGBoost Regressor"

                    else:

                        recommended_model = "LightGBM Regressor"

                # Unknown
                else:

                    recommended_model = "AutoML Model Selection Required"

                difficulty = "Easy"

                if summary["rows"] > 50000:
                    difficulty = "Medium"

                if summary["rows"] > 200000:
                    difficulty = "Hard"

                accuracy = "90 - 95%"

                if health_score < 70:
                    accuracy = "70 - 85%"

                elif health_score < 85:
                    accuracy = "85 - 90%"

                confidence = min(99, health_score + 5)

                tips = []

                if quality["missing_cells"] > 0:
                    tips.append(
                        "Fill missing values using Mean/Median for numerical features and Mode for categorical features."
                    )

                if quality["duplicate_rows"] > 0:
                    tips.append("Remove duplicate rows before model training.")

                if len(identifier_columns):
                    tips.append("Remove identifier columns from model training.")

                if summary["categorical"] > 0:
                    tips.append(
                        "Encode categorical variables using Label Encoding or One-Hot Encoding."
                    )

                if summary["numeric"] > 0:
                    tips.append(
                        "Scale numeric features before training distance-based algorithms."
                    )

                tips.append("Perform Feature Selection to improve model performance.")

                tips.append("Use Cross Validation for reliable evaluation.")

                tips.append("Hyperparameter tuning is recommended.")

                ai_report = {
                    "summary": (
                        f"IntelliDS analyzed {summary['rows']:,} records containing "
                        f"{summary['columns']} features. "
                        f"The dataset achieved a health score of {health_score}% and "
                        f"is classified as a {problem_type} problem. "
                        f"The engine recommends using {recommended_model} "
                        f"for optimal predictive performance."
                    ),
                    "problem_type": problem_type,
                    "model": recommended_model,
                    "difficulty": difficulty,
                    "accuracy": accuracy,
                    "confidence": confidence,
                    "tips": tips,
                }

            except Exception as e:

                form.add_error("dataset", f"Unable to read file : {e}")

            # ============================================================
            # DOWNLOAD PROFESSIONAL PDF REPORT
            # ============================================================

            if request.POST.get("action") == "pdf":

                from io import BytesIO
                from django.http import HttpResponse

                from reportlab.lib import colors
                from reportlab.lib.enums import TA_CENTER
                from reportlab.lib.styles import getSampleStyleSheet
                from reportlab.platypus import (
                    Paragraph,
                    SimpleDocTemplate,
                    Spacer,
                    Table,
                    TableStyle,
                    PageBreak,
                )

                from datetime import datetime

                buffer = BytesIO()

                doc = SimpleDocTemplate(
                    buffer,
                    rightMargin=35,
                    leftMargin=35,
                    topMargin=40,
                    bottomMargin=35,
                )

                styles = getSampleStyleSheet()

                title = styles["Heading1"]
                title.alignment = TA_CENTER

                heading = styles["Heading2"]

                normal = styles["BodyText"]

                story = []

                # ============================================================
                # TITLE
                # ============================================================

                story.append(Paragraph("IntelliDS Dataset Analysis Report", title))

                story.append(
                    Paragraph(f"Generated : {datetime.now():%d-%m-%Y %H:%M}", normal)
                )

                story.append(Spacer(1, 20))

                # ============================================================
                # DATASET SUMMARY
                # ============================================================

                story.append(Paragraph("Dataset Summary", heading))

                summary_table = Table(
                    [
                        ["Rows", summary["rows"]],
                        ["Columns", summary["columns"]],
                        ["Missing Values", summary["missing"]],
                        ["Duplicate Rows", summary["duplicates"]],
                        ["Memory Usage (MB)", summary["memory"]],
                        ["Numeric Columns", summary["numeric"]],
                        ["Categorical Columns", summary["categorical"]],
                        ["Datetime Columns", summary["datetime"]],
                        ["Boolean Columns", summary["boolean"]],
                    ],
                    colWidths=[220, 150],
                )

                summary_table.setStyle(
                    TableStyle(
                        [
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                            ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                        ]
                    )
                )

                story.append(summary_table)

                story.append(Spacer(1, 20))

                # ============================================================
                # DATASET HEALTH
                # ============================================================

                story.append(Paragraph("Dataset Health", heading))

                story.append(
                    Paragraph(f"<b>Health Score :</b> {health_score}%", normal)
                )

                story.append(Spacer(1, 15))

                # ============================================================
                # AI REPORT
                # ============================================================

                story.append(Paragraph("IntelliDS AI Recommendation", heading))

                story.append(Paragraph(ai_report.get("summary", ""), normal))

                ai_details = [
                    ["Problem Type", ai_report.get("problem_type", "")],
                    ["Recommended Model", ai_report.get("model", "")],
                    ["Difficulty", ai_report.get("difficulty", "")],
                    ["Expected Accuracy", ai_report.get("accuracy", "")],
                    ["Confidence", f"{ai_report.get('confidence',0)}%"],
                ]

                ai_table = Table(ai_details, colWidths=[150, 250])

                ai_table.setStyle(
                    TableStyle(
                        [
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                            ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                        ]
                    )
                )

                story.append(ai_table)

                story.append(Spacer(1, 20))

                # ============================================================
                # AI TIPS
                # ============================================================

                if ai_report.get("tips"):

                    story.append(Paragraph("AI Improvement Tips", heading))

                    for tip in ai_report["tips"]:

                        story.append(Paragraph(f"&bull; {tip}", normal))

                story.append(Spacer(1, 20))

                # ============================================================
                # RECOMMENDATIONS
                # ============================================================

                story.append(Paragraph("Recommendations", heading))

                for item in recommendations:

                    story.append(Paragraph(f"&bull; {item['message']}", normal))

                story.append(Spacer(1, 20))

                # ============================================================
                # MISSING VALUE REPORT
                # ============================================================

                if missing_columns:

                    story.append(Paragraph("Missing Value Analysis", heading))

                    missing_table = [["Column", "Missing Count", "Percentage"]]

                    for row in missing_columns:

                        missing_table.append(
                            [row["column"], row["missing"], f"{row['percent']}%"]
                        )

                    table = Table(missing_table, repeatRows=1)

                    table.setStyle(
                        TableStyle(
                            [
                                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                                (
                                    "BACKGROUND",
                                    (0, 0),
                                    (-1, 0),
                                    colors.HexColor("#1F4E78"),
                                ),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ]
                        )
                    )

                    story.append(table)

                story.append(PageBreak())

                # ============================================================
                # COLUMN PROFILE
                # ============================================================

                story.append(Paragraph("Column Profile", heading))

                profile_table = [["Column", "Type", "Missing", "Unique"]]

                for row in profile:

                    profile_table.append(
                        [row["name"], row["dtype"], row["missing"], row["unique"]]
                    )

                profile_tbl = Table(profile_table, repeatRows=1)

                profile_tbl.setStyle(
                    TableStyle(
                        [
                            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ]
                    )
                )

                story.append(profile_tbl)

                story.append(Spacer(1, 20))

                # ============================================================
                # NUMERIC STATISTICS
                # ============================================================

                if numeric_statistics:

                    story.append(Paragraph("Numeric Statistics", heading))

                    numeric_table = [["Column", "Mean", "Median", "Std", "Min", "Max"]]

                    for row in numeric_statistics:

                        numeric_table.append(
                            [
                                row["column"],
                                row["mean"],
                                row["median"],
                                row["std"],
                                row["min"],
                                row["max"],
                            ]
                        )

                    table = Table(numeric_table, repeatRows=1)

                    table.setStyle(
                        TableStyle(
                            [
                                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                                (
                                    "BACKGROUND",
                                    (0, 0),
                                    (-1, 0),
                                    colors.HexColor("#1F4E78"),
                                ),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ]
                        )
                    )

                    story.append(table)

                # ============================================================
                # CATEGORICAL STATISTICS
                # ============================================================

                if categorical_statistics:

                    story.append(Spacer(1, 20))

                    story.append(Paragraph("Categorical Statistics", heading))

                    cat_table = [["Column", "Unique", "Top Value", "Frequency"]]

                    for row in categorical_statistics:

                        cat_table.append(
                            [row["column"], row["unique"], row["top"], row["frequency"]]
                        )

                    table = Table(cat_table, repeatRows=1)

                    table.setStyle(
                        TableStyle(
                            [
                                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                                (
                                    "BACKGROUND",
                                    (0, 0),
                                    (-1, 0),
                                    colors.HexColor("#1F4E78"),
                                ),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ]
                        )
                    )

                    story.append(table)

                # ============================================================
                # FOOTER
                # ============================================================

                story.append(Spacer(1, 30))

                story.append(
                    Paragraph(
                        "<i>Generated automatically by IntelliDS AutoML Platform</i>",
                        styles["Italic"],
                    )
                )

                doc.build(story)

                pdf = buffer.getvalue()

                buffer.close()

                response = HttpResponse(pdf, content_type="application/pdf")

                response["Content-Disposition"] = (
                    'attachment; filename="IntelliDS_Dataset_Report.pdf"'
                )

                return response

    context = {
        "form": form,
        "preview": preview,
        "columns": columns,
        "summary": summary,
        "profile": profile,
        "quality": quality,
        "health_score": health_score,
        "identifier_columns": identifier_columns,
        "missing_columns": missing_columns,
        "categorical_statistics": categorical_statistics,
        "column_categories": column_categories,
        "numeric_statistics": numeric_statistics,
        "dataset_ready": dataset_ready,
        "ai_report": ai_report,
        "target_candidates": target_candidates,
        "recommendations": recommendations,
    }

    return render(
        request,
        "core/intellids.html",
        context,
    )


# ================================================================================================================================

INTEGER_KEYWORDS = [
    "age",
    "year",
    "month",
    "day",
    "count",
    "number",
    "qty",
    "quantity",
    "children",
    "kids",
    "rooms",
    "rank",
    "semester",
    "attempt",
    "marks",
    "score",
    "orders",
    "purchases",
    "visits",
    "employees",
    "students",
    "patients",
    "tickets",
    "credits",
    "units",
    "experience",
    "yrs",
    "years",
]


def clean_integer_columns(df):
    """
    Clean columns that should contain only integers.

    Rules
    -----
    • 34.0 -> 34
    • 34.5 -> NaN
    • 0.34 -> NaN
    • -2.7 -> NaN
    """

    cleaned = []

    for col in df.columns:

        col_name = col.lower()

        if not any(keyword in col_name for keyword in INTEGER_KEYWORDS):
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        series = df[col]

        # Values having decimal part
        decimal_mask = series.notna() & (~np.isclose(series % 1, 0))

        if decimal_mask.any():

            df.loc[decimal_mask, col] = np.nan

            cleaned.append(col)

    return df, cleaned


def restore_integer_columns(df):
    """
    Restore float columns back to nullable integers whenever appropriate.

    Rules:
    1. If every non-null value is an integer -> convert.
    2. If the column name suggests integer values (Age, Count, Qty, etc.)
       and values are almost integers, round and convert.
    """

    restored = []

    float_cols = df.select_dtypes(include=["float"]).columns

    for col in float_cols:

        series = df[col]

        non_null = series.dropna()

        if non_null.empty:
            continue

        col_name = col.lower()

        # ------------------------------------
        # Rule 1
        # ------------------------------------
        if np.all(np.isclose(non_null % 1, 0)):

            df[col] = series.astype("Int64")

            restored.append(col)

            continue

        # ------------------------------------
        # Rule 2
        # ------------------------------------
        if any(keyword in col_name for keyword in INTEGER_KEYWORDS):

            decimal_distance = np.abs(non_null - np.round(non_null))

            if (decimal_distance < 0.001).all():

                df[col] = np.round(series).astype("Int64")

                restored.append(col)

    return df, restored


# ==============================================================================
# Convert Numeric Strings
# ==============================================================================


def convert_numeric_strings(df):
    """
    Convert object columns containing numeric values into numeric dtype.
    """

    object_cols = df.select_dtypes(include=["object"]).columns

    converted = []

    for col in object_cols:

        numeric = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        if numeric.notna().sum() >= len(df) * 0.8:

            df[col] = numeric

            converted.append(col)

    return df, converted


# ==============================================================================
# Clean String Values
# ==============================================================================


def clean_string_values(df):
    """
    Trim spaces and remove leading/trailing special characters
    without affecting missing values.
    """

    object_cols = df.select_dtypes(include=["object"]).columns

    for col in object_cols:

        mask = df[col].notna()

        df.loc[mask, col] = (
            df.loc[mask, col]
            .astype(str)
            .str.strip()
            .str.replace(
                r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$",
                "",
                regex=True,
            )
        )

    return df


# ==============================================================================
# Replace Infinite Values
# ==============================================================================


def replace_infinite_values(df):
    """
    Replace positive and negative infinity with NaN.
    """

    count = np.isinf(df.select_dtypes(include=["number"])).sum().sum()

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    return df, int(count)


# ==============================================================================
# Detect Duplicate Columns
# ==============================================================================


def find_duplicate_columns(df):
    """
    Detect duplicate columns containing identical values.
    """

    duplicates = []

    cols = df.columns.tolist()

    for i in range(len(cols)):

        for j in range(i + 1, len(cols)):

            if df[cols[i]].equals(df[cols[j]]):

                duplicates.append(cols[j])

    return duplicates


# ==============================================================================
# Standardize Text Case
# ==============================================================================


def standardize_text_case(df):
    """
    Standardize categorical text values.

    Examples:
        male      -> Male
        MALE      -> Male
        mumbai    -> Mumbai
        NEW YORK  -> New York

    Skips columns that appear to be IDs or codes.
    """

    object_cols = df.select_dtypes(include=["object"]).columns

    standardized = []

    for col in object_cols:

        # Skip identifier-like columns
        col_name = col.lower()

        if any(
            keyword in col_name
            for keyword in [
                "id",
                "code",
                "uuid",
                "pin",
                "zip",
                "postal",
                "account",
                "mobile",
                "phone",
                "email",
            ]
        ):
            continue

        mask = df[col].notna()

        before = df.loc[mask, col].copy()

        df.loc[mask, col] = df.loc[mask, col].astype(str).str.strip().str.title()

        if not before.equals(df.loc[mask, col]):

            standardized.append(col)

    return df, standardized


# ==============================================================================
# Treat Outliers (IQR Method)
# ==============================================================================


def treat_outliers_iqr(df):
    """
    Detect and cap numeric outliers using the IQR method.

    Skips:
    - ID columns
    - Date/Year columns
    - Encoded categories
    - Ratings/Scores
    - Flags and status columns
    """

    SKIP_OUTLIER_KEYWORDS = [
        # -------------------------
        # Identifier columns
        # -------------------------
        "id",
        "code",
        "number",
        "no",
        "uuid",
        "increment",
        "account",
        "customer",
        "merchant",
        "transaction",
        "order",
        "student",
        "product",
        "white",
        "black",
        "show",
        # -------------------------
        # Date / Time columns
        # -------------------------
        "date",
        "time",
        "timestamp",
        "created",
        "updated",
        "move",
        "month",
        "day",
        "hour",
        # -------------------------
        # Year columns
        # Matches:
        # year
        # Year
        # release_year
        # Release_Years
        # model_year
        # -------------------------
        "year",
        "years",
        "release",
        "generation",
        # -------------------------
        # Category / Label columns
        # -------------------------
        "type",
        "status",
        "class",
        "category",
        "region",
        "location",
        "country",
        "fuel",
        "owner",
        "seller",
        "transmission",
        "rating",
        # -------------------------
        # Scores / Ratings
        # -------------------------
        "score",
        "scores",
        "quality",
        "grade",
        "rank",
        "level",
        "victory",
        "winner",
        # -------------------------
        # Boolean / Flag columns
        # -------------------------
        "is_",
        "has_",
        "used",
        "foreign",
        "fraud",
        "scam",
        "vpn",
        "mismatch",
        # -------------------------
        # Encoded / Game data
        # -------------------------
        "generation",
        "opening",
        "ply",
        # -------------------------
        # Bounded values
        # -------------------------
        "percent",
        "percentage",
        "rate",
        "rating",
        "probability",
        "velocity",
        "engine_size",
        "mileage",
        "horsepower",
        "torque",
        "selling_price",
    ]

    numeric_cols = df.select_dtypes(include=["number"]).columns

    treated = {}

    for col in numeric_cols:

        col_name = col.lower().replace(" ", "_")

        # Skip unsafe columns
        if any(keyword in col_name for keyword in SKIP_OUTLIER_KEYWORDS):
            continue

        series = df[col]

        # Skip low variation columns
        if series.nunique() < 5:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - (1.5 * iqr)

        upper = q3 + (1.5 * iqr)

        mask = (series < lower) | (series > upper)

        count = int(mask.sum())

        if count:

            df[col] = series.clip(lower, upper)

            treated[col] = count

    return df, treated


# ==============================================================================
# Apply Cleaning Pipeline
# ==============================================================================


def apply_cleaning_pipeline(df, options=None):
    """
    Apply data cleaning pipeline.
    Returns:
        {
            "dataframe": cleaned_df,
            "summary": cleaned_summary,
            "report": cleaning_report,
        }
    """

    if options is None:
        options = default_cleaning_options()

    cleaned_df = df.copy()

    report = {
        "steps": [],
        "rows_before": len(df),
        "columns_before": len(df.columns),
    }

    # --------------------------------------------------------------------------
    # Fix Column Names
    # --------------------------------------------------------------------------

    if options.get("fix_column_names", True):

        old_columns = cleaned_df.columns.tolist()

        cleaned_df.columns = [
            re.sub(
                "_+", "_", str(col).strip().replace(" ", "_").replace("-", "_")
            ).strip("_")
            for col in cleaned_df.columns
        ]

        if old_columns != cleaned_df.columns.tolist():

            report["steps"].append("Column names standardized.")

    # --------------------------------------------------------------------------
    # Clean String Columns
    # --------------------------------------------------------------------------

    if options.get("trim_strings", True):

        cleaned_df = clean_string_values(cleaned_df)

        report["steps"].append(
            "Trimmed spaces and removed unwanted symbols from text values."
        )

    # --------------------------------------------------------------------------
    # Standardize Missing Values
    # --------------------------------------------------------------------------

    if options.get("standardize_missing", True):

        cleaned_df.replace(
            [
                "",
                " ",
                "NA",
                "N/A",
                "NULL",
                "null",
                "None",
                "none",
                "?",
                "-",
            ],
            np.nan,
            inplace=True,
        )

        report["steps"].append("Standardized missing values.")

    # --------------------------------------------------------------------------
    # Convert Numeric Strings
    # --------------------------------------------------------------------------

    if options.get("convert_numeric_strings", True):

        cleaned_df, converted = convert_numeric_strings(cleaned_df)

        if converted:

            report["numeric_columns_detected"] = converted

            report["steps"].append(
                f"Converted {len(converted)} text column(s) into numeric."
            )

    # --------------------------------------------------------------------------
    # Replace Infinite Values
    # --------------------------------------------------------------------------

    if options.get("replace_infinite_values", True):

        cleaned_df, count = replace_infinite_values(cleaned_df)

        if count:

            report["infinite_values_replaced"] = count

            report["steps"].append(
                f"Replaced {count} infinite value(s) with missing values."
            )

    # --------------------------------------------------------------------------
    # Clean Integer Columns
    # --------------------------------------------------------------------------

    if options.get("clean_integer_columns", True):

        cleaned_df, cleaned = clean_integer_columns(cleaned_df)

        if cleaned:

            report["integer_columns_cleaned"] = cleaned

            report["steps"].append(
                f"Corrected invalid decimal values in {len(cleaned)} integer column(s)."
            )

    # --------------------------------------------------------------------------
    # Remove Empty Rows
    # --------------------------------------------------------------------------

    if options.get("remove_empty_rows", True):

        before = len(cleaned_df)

        cleaned_df.dropna(
            how="all",
            inplace=True,
        )

        removed = before - len(cleaned_df)

        if removed:

            report["empty_rows_removed"] = removed

            report["steps"].append(f"Removed {removed} completely empty row(s).")

    # --------------------------------------------------------------------------
    # Remove Duplicate Rows
    # --------------------------------------------------------------------------

    if options.get("remove_duplicates", True):

        before = len(cleaned_df)

        cleaned_df.drop_duplicates(inplace=True)

        removed = before - len(cleaned_df)

        report["duplicates_removed"] = int(removed)

        report["steps"].append(f"Removed {removed} duplicate rows.")

    # --------------------------------------------------------------------------
    # Fill Numeric Missing
    # --------------------------------------------------------------------------

    if options.get("fill_numeric_missing", True):

        numeric_cols = cleaned_df.select_dtypes(include=["number"]).columns

        for col in numeric_cols:

            if cleaned_df[col].isna().any():

                col_name = col.lower()

                if any(keyword in col_name for keyword in INTEGER_KEYWORDS):

                    mode = cleaned_df[col].mode()

                    if not mode.empty:
                        cleaned_df[col] = cleaned_df[col].fillna(mode.iloc[0])

                else:

                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())

        report["steps"].append("Filled numeric missing values.")

    # --------------------------------------------------------------------------
    # Treat Outliers (IQR)
    # --------------------------------------------------------------------------

    if options.get("treat_outliers", True):

        cleaned_df, treated = treat_outliers_iqr(cleaned_df)

        if treated:

            report["outliers_treated"] = treated

            total = sum(treated.values())

            report["steps"].append(
                f"Treated {total} outlier value(s) across {len(treated)} numeric column(s)."
            )

    # --------------------------------------------------------------------------
    # Fill Categorical Missing
    # --------------------------------------------------------------------------

    if options.get("fill_categorical_missing", True):

        object_cols = cleaned_df.select_dtypes(include=["object"]).columns

        for col in object_cols:

            if cleaned_df[col].isna().any():

                mode = cleaned_df[col].mode()

                value = mode.iloc[0] if not mode.empty else "Unknown"

                cleaned_df[col] = cleaned_df[col].fillna(value)

        report["steps"].append("Filled categorical missing values.")

    # --------------------------------------------------------------------------
    # Restore Integer Columns
    # --------------------------------------------------------------------------

    if options.get("restore_integer_columns", True):

        cleaned_df, restored = restore_integer_columns(cleaned_df)

        if restored:

            report["integer_columns_restored"] = restored

            report["steps"].append(
                f"Restored {len(restored)} numeric column(s) back to integer."
            )

    # --------------------------------------------------------------------------
    # Standardize Text Case
    # --------------------------------------------------------------------------

    if options.get("standardize_text_case", True):

        cleaned_df, standardized_cols = standardize_text_case(cleaned_df)

        if standardized_cols:

            report["text_case_standardized"] = standardized_cols

            report["steps"].append(
                f"Standardized text capitalization in {len(standardized_cols)} column(s)."
            )

    # --------------------------------------------------------------------------
    # Remove Constant Columns
    # --------------------------------------------------------------------------

    if options.get("remove_constant_columns", True):

        constant_cols = [
            col
            for col in cleaned_df.columns
            if cleaned_df[col].nunique(dropna=False) <= 1
        ]

        if constant_cols:

            cleaned_df.drop(
                columns=constant_cols,
                inplace=True,
            )

            report["constant_columns_removed"] = constant_cols

            report["steps"].append(f"Removed {len(constant_cols)} constant columns.")

    # --------------------------------------------------------------------------
    # Remove Duplicate Columns
    # --------------------------------------------------------------------------

    if options.get("remove_duplicate_columns", True):

        duplicate_cols = find_duplicate_columns(cleaned_df)

        if duplicate_cols:

            cleaned_df.drop(
                columns=duplicate_cols,
                inplace=True,
            )

            report["duplicate_columns_removed"] = duplicate_cols

            report["steps"].append(
                f"Removed {len(duplicate_cols)} duplicate column(s)."
            )

    # --------------------------------------------------------------------------
    # Convert Boolean Columns
    # --------------------------------------------------------------------------

    if options.get("convert_booleans", True):

        bool_cols = cleaned_df.select_dtypes(include=["bool"]).columns

        for col in bool_cols:

            cleaned_df[col] = cleaned_df[col].astype(int)

        if len(bool_cols):

            report["steps"].append("Converted boolean columns.")

    # --------------------------------------------------------------------------
    # Convert Date Columns
    # --------------------------------------------------------------------------

    if options.get("convert_dates", True):

        for col in cleaned_df.columns:

            if cleaned_df[col].dtype == object:

                try:

                    converted = pd.to_datetime(
                        cleaned_df[col],
                        errors="coerce",
                    )

                    if converted.notna().sum() > len(cleaned_df) * 0.8:

                        cleaned_df[col] = converted

                except Exception:

                    pass

        report["steps"].append("Attempted datetime conversion.")

    # --------------------------------------------------------------------------
    # Remove High Missing Columns
    # --------------------------------------------------------------------------

    if options.get("remove_high_missing_columns", False):

        threshold = options.get(
            "high_missing_threshold",
            0.80,
        )

        remove_cols = []

        for col in cleaned_df.columns:

            missing_ratio = cleaned_df[col].isna().mean()

            if missing_ratio >= threshold:

                remove_cols.append(col)

        if remove_cols:

            cleaned_df.drop(
                columns=remove_cols,
                inplace=True,
            )

            report["high_missing_columns_removed"] = remove_cols

            report["steps"].append(f"Removed {len(remove_cols)} high-missing columns.")

    # --------------------------------------------------------------------------
    # Final Summary
    # --------------------------------------------------------------------------

    report["rows_after"] = len(cleaned_df)

    report["columns_after"] = len(cleaned_df.columns)

    cleaned_summary = generate_summary(cleaned_df)

    return {
        "dataframe": cleaned_df,
        "summary": cleaned_summary,
        "report": report,
    }


# ==============================================================================
# Dataset Summary
# ==============================================================================


def generate_summary(df):
    """
    Generate dataset summary.
    """

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(
            df.memory_usage(deep=True).sum() / (1024 * 1024),
            2,
        ),
        "numeric_columns": len(df.select_dtypes(include=["number"]).columns),
        "categorical_columns": len(df.select_dtypes(include=["object"]).columns),
        "datetime_columns": len(df.select_dtypes(include=["datetime"]).columns),
        "boolean_columns": len(df.select_dtypes(include=["bool"]).columns),
    }


# ==============================================================================
# Default Cleaning Options
# ==============================================================================


def default_cleaning_options():
    """
    Default cleaning pipeline options.
    """

    return {
        "trim_strings": True,
        "fix_column_names": True,
        "standardize_missing": True,
        "remove_duplicates": True,
        "fill_numeric_missing": True,
        "fill_categorical_missing": True,
        "remove_constant_columns": True,
        "convert_dates": True,
        "convert_booleans": True,
        "detect_identifiers": True,
        "remove_high_missing_columns": False,
        "high_missing_threshold": 0.80,
        "convert_numeric_strings": True,
        "remove_empty_rows": True,
        "replace_infinite_values": True,
        "remove_duplicate_columns": True,
        "standardize_text_case": True,
        "treat_outliers": True,
        "restore_integer_columns": True,
        "clean_integer_columns": True,
    }


@login_required
def data_preparation(request):
    """
    IntelliDS AI
    Automatic Data Preparation Pipeline

    Upload Dataset
    ->
    Clean Automatically
    ->
    Preview
    ->
    Download XLSX
    """

    form = DatasetUploadForm()

    dataset_loaded = False

    preview = []
    cleaned_preview = []

    columns = []

    summary = {}
    cleaned_summary = {}

    # preprocessing = {}

    cleaning_report = {}

    download_url = None

    if request.method == "POST":

        # ==========================================================
        # MANUAL COLUMN DROP
        # ==========================================================

        if request.POST.get("action") == "drop_columns":

            json_data = request.session.get("cleaned_dataset")

            if json_data:

                cleaned_df = pd.read_json(
                    StringIO(json_data),
                    orient="split",
                )

                columns_to_drop = request.POST.getlist("drop_columns")

                cleaned_df.drop(
                    columns=columns_to_drop,
                    inplace=True,
                    errors="ignore",
                )

                export_dir = os.path.join(
                    settings.MEDIA_ROOT,
                    "cleaned_datasets",
                )

                os.makedirs(export_dir, exist_ok=True)

                filename = "Final_Cleaned_Dataset.xlsx"

                filepath = os.path.join(
                    export_dir,
                    filename,
                )

                cleaned_df.to_excel(
                    filepath,
                    index=False,
                )

                download_url = settings.MEDIA_URL + "cleaned_datasets/" + filename

                cleaned_preview = (
                    cleaned_df.head(50).fillna("").astype(str).values.tolist()
                )

                columns = cleaned_df.columns.tolist()

                cleaned_summary = generate_summary(cleaned_df)

                dataset_loaded = True

                context = {
                    "form": DatasetUploadForm(),
                    "dataset_loaded": dataset_loaded,
                    "summary": cleaned_summary,
                    "cleaned_summary": cleaned_summary,
                    "columns": columns,
                    "preview": cleaned_preview,
                    "cleaned_preview": cleaned_preview,
                    "cleaning_report": {
                        "steps": [f"User removed {len(columns_to_drop)} column(s)."]
                    },
                    "download_url": download_url,
                }

                return render(
                    request,
                    "core/data_preparation.html",
                    context,
                )

        form = DatasetUploadForm(request.POST, request.FILES)

        if form.is_valid():

            try:

                uploaded_file = form.cleaned_data["dataset"]

                # ==========================================================
                # READ DATASET
                # ==========================================================

                df = read_dataset(uploaded_file)

                dataset_loaded = True

                # ==========================================================
                # ORIGINAL DATASET INFO
                # ==========================================================

                df.columns = [str(col).strip() for col in df.columns]

                summary = generate_summary(df)

                # preprocessing = detect_feature_types(df)

                columns = df.columns.tolist()

                preview = df.head(200).fillna("").astype(str).values.tolist()

                # ==========================================================
                # AUTOMATIC CLEANING
                # ==========================================================

                cleaning_options = default_cleaning_options()

                result = apply_cleaning_pipeline(df, cleaning_options)

                cleaned_df = result["dataframe"]
                # Save cleaned dataframe in session
                request.session["cleaned_dataset"] = cleaned_df.to_json(
                    orient="split", date_format="iso"
                )

                cleaned_summary = result["summary"]

                cleaning_report = result["report"]

                # ==========================================================
                # CLEANED PREVIEW
                # ==========================================================

                cleaned_preview = (
                    cleaned_df.head(200).fillna("").astype(str).values.tolist()
                )

                # ==========================================================
                # SAVE CLEANED XLSX
                # ==========================================================

                export_dir = os.path.join(settings.MEDIA_ROOT, "cleaned_datasets")

                os.makedirs(export_dir, exist_ok=True)

                filename = os.path.splitext(uploaded_file.name)[0] + "_Prepared.xlsx"

                filepath = os.path.join(export_dir, filename)

                cleaned_df.to_excel(filepath, index=False)

                # ==========================================================
                # SAVE REPORT
                # ==========================================================

                report_filename = (
                    os.path.splitext(uploaded_file.name)[0] + "_Cleaning_Report.json"
                )

                report_path = os.path.join(export_dir, report_filename)

                with open(report_path, "w", encoding="utf-8") as f:

                    json.dump(cleaning_report, f, indent=4, default=str)

                # ==========================================================
                # DOWNLOAD URL
                # ==========================================================

                # download_url = settings.MEDIA_URL + "cleaned_datasets/" + filename
                download_url = settings.MEDIA_URL + "cleaned_datasets/" + filename

            except Exception as e:

                form.add_error("dataset", str(e))

    context = {
        "form": form,
        "dataset_loaded": dataset_loaded,
        "summary": summary,
        "cleaned_summary": cleaned_summary,
        # "preprocessing": preprocessing,
        "columns": columns,
        "preview": preview,
        "cleaned_preview": cleaned_preview,
        "cleaning_report": cleaning_report,
        "download_url": download_url,
    }

    return render(request, "core/data_preparation.html", context)


# ==============================================================================================================================


@login_required
def model_training(request):

    # =====================================================
    # RESET DATASET
    # =====================================================

    if request.GET.get("reset") == "1":

        if request.GET.get("reset") == "1":

            for key in [
                "ml_dataset",
                "ml_columns",
                "ml_features",
                "ml_model_results",
                "ml_best_model",
                "ml_graph_url",
                "ml_problem_type",
            ]:
                request.session.pop(key, None)

            model_path = os.path.join(
                settings.MEDIA_ROOT,
                "ml_models",
                "model_package.pkl",
            )

            if os.path.exists(model_path):
                os.remove(model_path)

            return redirect("model_training")

    # =====================================================
    # INITIAL VALUES
    # =====================================================

    form = DatasetUploadForm()

    prediction = None

    problem_type = request.session.get("ml_problem_type", None)

    columns = request.session.get("ml_columns", [])

    dataset_loaded = bool(columns)

    summary = {}
    preview_data = []
    preview_columns = []

    model_results = request.session.get("ml_model_results", [])

    best_model = request.session.get("ml_best_model", {})

    model_explanation = request.session.get("ml_model_explanation", "")

    selected_model_name = ""

    if best_model:
        selected_model_name = best_model.get("model", "")

    graph_url = request.session.get("ml_graph_url")

    prediction_fields = request.session.get("ml_features", [])

    temp_file = request.session.get("ml_dataset")

    if temp_file and os.path.exists(temp_file):

        df = pd.read_pickle(temp_file)

        summary = {
            "rows": df.shape[0],
            "columns": df.shape[1],
        }

        preview_columns = df.columns.tolist()
        preview_data = df.head(100).fillna("").values.tolist()

    # =====================================================
    # HANDLE POST REQUESTS
    # =====================================================

    if request.method == "POST":

        # =====================================================
        # STEP 1 : UPLOAD DATASET
        # =====================================================

        if "dataset" in request.FILES:

            form = DatasetUploadForm(request.POST, request.FILES)

            if form.is_valid():

                try:

                    uploaded_file = form.cleaned_data["dataset"]

                    df = read_dataset(uploaded_file)

                    columns = df.columns.tolist()

                    dataset_loaded = True

                    summary = {
                        "rows": df.shape[0],
                        "columns": df.shape[1],
                    }

                    preview_columns = df.columns.tolist()
                    preview_data = df.head(100).fillna("").values.tolist()

                    request.session["ml_columns"] = columns

                    temp_dir = os.path.join(
                        settings.MEDIA_ROOT,
                        "temp_ml",
                    )

                    os.makedirs(temp_dir, exist_ok=True)

                    temp_file = os.path.join(
                        temp_dir,
                        uploaded_file.name,
                    )

                    df.to_pickle(temp_file)

                    request.session["ml_dataset"] = temp_file

                    # Clear previous model information
                    request.session.pop("ml_features", None)
                    request.session.pop("ml_model_results", None)
                    request.session.pop("ml_best_model", None)
                    request.session.pop("ml_graph_url", None)

                except Exception as e:

                    form.add_error("dataset", str(e))

        # =====================================================
        # STEP 2 : TRAIN MODEL
        # =====================================================

        elif request.POST.get("target_column"):

            target_column = request.POST.get("target_column")

            temp_file = request.session.get("ml_dataset")

            if temp_file and target_column:

                df = pd.read_pickle(temp_file)

                columns = df.columns.tolist()

                # ============================
                # Separate Features & Target
                # ============================

                X = df.drop(columns=[target_column])

                y = df[target_column]

                # -------------------------------------------------
                # Remove identifier columns (Name, ID, etc.)
                # -------------------------------------------------

                # IDENTIFIER_WORDS = [
                #     "id",
                #     "name",
                #     "customer",
                #     "account",
                #     "mobile",
                #     "phone",
                #     "email",
                #     "address",
                #     "pan",
                #     "aadhaar",
                #     "passport",
                #     "uuid",
                # ]

                # drop_cols = []

                # drop_cols = []

                # for col in X.columns:

                #     column_name = col.lower()

                #     if any(word in column_name for word in IDENTIFIER_WORDS):
                #         drop_cols.append(col)

                # # for col in X.columns:

                # #     if X[col].dtype == "object":

                # #         unique_ratio = X[col].nunique() / len(X)

                # #         if any(word in column_name for word in IDENTIFIER_WORDS):
                # #             drop_cols.append(col)

                # #         # if unique_ratio > 0.80:
                # #         #     drop_cols.append(col)

                # X = X.drop(columns=drop_cols)

                # print("Dropped Identifier Columns :", drop_cols)

                # Save ONLY feature columns for prediction
                prediction_fields = X.columns.tolist()
                request.session["ml_features"] = prediction_fields

                # ============================
                # Feature Preprocessing
                # ============================

                for col in X.columns:

                    if pd.api.types.is_datetime64_any_dtype(X[col]):

                        X[col] = X[col].astype("int64")

                categorical_columns = X.select_dtypes(
                    include=["object", "category", "string"]
                ).columns

                encoders = {}

                for col in categorical_columns:

                    encoder = LabelEncoder()

                    X[col] = encoder.fit_transform(X[col].astype(str))

                    encoders[col] = encoder

                for col in X.columns:

                    X[col] = pd.to_numeric(
                        X[col],
                        errors="coerce",
                    )

                X = X.fillna(0)

                X = X.replace([np.inf, -np.inf], 0)

                scaler = StandardScaler()

                X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

                # =====================================================
                # AUTO DETECT PROBLEM TYPE
                # =====================================================

                target_encoder = None

                unique_count = y.nunique()
                unique_ratio = unique_count / len(y)

                problem_type = "classification"

                # Numeric target -> Regression
                if pd.api.types.is_numeric_dtype(y):
                    if unique_count > 20:
                        problem_type = "regression"

                # Object/String target
                else:

                    # Try converting values like ₹10,000, €20M etc.
                    y_numeric = (
                        y.astype(str)
                        .str.replace(",", "", regex=False)
                        .str.replace("₹", "", regex=False)
                        .str.replace("$", "", regex=False)
                        .str.replace("€", "", regex=False)
                        .str.replace("M", "", regex=False)
                        .str.replace("K", "", regex=False)
                    )

                    converted = pd.to_numeric(y_numeric, errors="coerce")

                    # If mostly numeric after cleaning -> Regression
                    if converted.notna().mean() > 0.90:
                        y = converted
                        problem_type = "regression"

                    # Too many unique categories -> Regression
                    elif unique_count > 30 or unique_ratio > 0.30:
                        y = pd.factorize(y)[0].astype(float)
                        problem_type = "regression"

                # Classification Encoding
                if problem_type == "classification":

                    target_encoder = LabelEncoder()
                    y = target_encoder.fit_transform(y.astype(str))

                print("=" * 60)
                print("Problem Type :", problem_type)
                print("Target Unique :", unique_count)
                print("=" * 60)

                request.session["ml_problem_type"] = problem_type

                # =====================================================
                # MODEL EXPLANATION GENERATION
                # =====================================================

                model_explanation = ""

                if problem_type == "classification":

                    original_classes = df[target_column].nunique()

                    final_classes = len(np.unique(y))

                    if original_classes > final_classes:

                        model_explanation = (
                            f"Initially, {target_column} prediction was treated as a "
                            f"multi-class classification problem using {original_classes} "
                            f"original classes. Due to class imbalance and variations in "
                            f"target categories, model performance was limited. "
                            f"The target variable was transformed into {final_classes} "
                            f"meaningful categories, improving class balance and allowing "
                            f"machine learning models to learn better patterns. "
                            f"After this transformation, the models achieved improved "
                            f"classification performance."
                        )

                    else:

                        model_explanation = (
                            f"The dataset was analyzed as a classification problem with "
                            f"{final_classes} target categories. Machine learning models "
                            f"were trained using preprocessing, feature transformation, "
                            f"and multiple classification algorithms to identify patterns "
                            f"in the target variable."
                        )

                else:

                    model_explanation = (
                        f"The dataset was analyzed as a regression problem where "
                        f"{target_column} was treated as a continuous numerical target. "
                        f"Multiple regression algorithms were trained and evaluated "
                        f"using R2 Score, MAE, RMSE, and MSE metrics."
                    )

                request.session["ml_model_explanation"] = model_explanation

                # =====================================================
                # TRAIN TEST SPLIT
                # =====================================================

                if problem_type == "classification":

                    class_counts = pd.Series(y).value_counts()

                    # Use stratify only when every class has at least 2 samples
                    if class_counts.min() >= 2:
                        stratify = y
                    else:
                        stratify = None

                else:
                    stratify = None

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.20,
                    random_state=42,
                    stratify=stratify,
                )

                # ============================
                # Models
                # ============================

                if problem_type == "classification":

                    models = {
                        "Random Forest": RandomForestClassifier(
                            n_estimators=300, random_state=42
                        ),
                        "Extra Trees": ExtraTreesClassifier(
                            n_estimators=300, random_state=42
                        ),
                        "Gradient Boosting": GradientBoostingClassifier(),
                        "AdaBoost": AdaBoostClassifier(),
                        "Logistic Regression": LogisticRegression(max_iter=5000),
                        "Decision Tree": DecisionTreeClassifier(),
                        "KNN": KNeighborsClassifier(),
                        "SVM": SVC(),
                    }

                else:

                    models = {
                        "Random Forest": RandomForestRegressor(
                            n_estimators=300, random_state=42
                        ),
                        "Extra Trees": ExtraTreesRegressor(
                            n_estimators=300, random_state=42
                        ),
                        "Gradient Boosting": GradientBoostingRegressor(),
                        "AdaBoost": AdaBoostRegressor(),
                        "Linear Regression": LinearRegression(),
                        "Decision Tree": DecisionTreeRegressor(),
                        "KNN": KNeighborsRegressor(),
                        "SVR": SVR(),
                    }

                model_results = []

                trained_models = {}

                for name, model in models.items():

                    model.fit(X_train, y_train)

                    pred = model.predict(X_test)

                    trained_models[name] = model

                    if problem_type == "classification":

                        model_results.append(
                            {
                                "model": name,
                                "accuracy": round(
                                    accuracy_score(y_test, pred) * 100, 2
                                ),
                                "precision": round(
                                    precision_score(
                                        y_test,
                                        pred,
                                        average="weighted",
                                        zero_division=0,
                                    )
                                    * 100,
                                    2,
                                ),
                                "recall": round(
                                    recall_score(
                                        y_test,
                                        pred,
                                        average="weighted",
                                        zero_division=0,
                                    )
                                    * 100,
                                    2,
                                ),
                                "f1": round(
                                    f1_score(
                                        y_test,
                                        pred,
                                        average="weighted",
                                        zero_division=0,
                                    )
                                    * 100,
                                    2,
                                ),
                            }
                        )

                    else:

                        model_results.append(
                            {
                                "model": name,
                                "r2_score": round(r2_score(y_test, pred) * 100, 2),
                                "mae": round(mean_absolute_error(y_test, pred), 4),
                                "rmse": round(
                                    np.sqrt(mean_squared_error(y_test, pred)), 4
                                ),
                                "mse": round(mean_squared_error(y_test, pred), 4),
                            }
                        )

                # ============================
                # Best Model
                # ============================

                # ============================
                # Best Model + Graph Sorting
                # ============================

                if problem_type == "classification":

                    best_model = max(model_results, key=lambda x: x["accuracy"])

                    graph_results = sorted(
                        model_results, key=lambda x: x["accuracy"], reverse=True
                    )

                else:

                    best_model = max(model_results, key=lambda x: x["r2_score"])

                    graph_results = sorted(
                        model_results, key=lambda x: x["r2_score"], reverse=True
                    )

                print("BEST MODEL :", best_model)

                # best_model = model_results[0]

                # ============================
                # Save Model
                # ============================

                model_dir = os.path.join(
                    settings.MEDIA_ROOT,
                    "ml_models",
                )

                os.makedirs(model_dir, exist_ok=True)

                model_package = {
                    "models": trained_models,
                    "best_model": best_model["model"],
                    "features": X.columns.tolist(),
                    "target_column": target_column,
                    "encoders": encoders,
                    "scaler": scaler,
                    "target_encoder": target_encoder,
                    "feature_types": {col: str(df[col].dtype) for col in X.columns},
                    # "dropped_columns": drop_cols,
                }

                with open(
                    os.path.join(
                        model_dir,
                        "model_package.pkl",
                    ),
                    "wb",
                ) as f:

                    pickle.dump(model_package, f)

                # ============================
                # Accuracy Graph
                # ============================

                graph_dir = os.path.join(
                    settings.MEDIA_ROOT,
                    "ml_graphs",
                )

                os.makedirs(graph_dir, exist_ok=True)

                # if problem_type == "classification":

                #     model_results = sorted(
                #         model_results, key=lambda x: x["accuracy"], reverse=True
                #     )

                # else:

                #     model_results = sorted(
                #         model_results, key=lambda x: x["r2_score"], reverse=True
                #     )

                models = [x["model"] for x in graph_results]
                if problem_type == "classification":

                    graph_values = [x["accuracy"] for x in graph_results]

                    graph_label = "Accuracy (%)"

                else:

                    graph_values = [x["r2_score"] for x in graph_results]

                    graph_label = "R2 Score (%)"

                plt.figure(figsize=(12, 6))

                colors = []

                for m in graph_results:

                    if m["model"] == best_model["model"]:

                        colors.append("green")

                    else:

                        colors.append("steelblue")

                bars = plt.barh(
                    models,
                    graph_values,
                    color=colors,
                )

                plt.xlim(0, 100)

                # plt.xlabel("Accuracy (%)", fontsize=12)
                plt.xlabel(graph_label, fontsize=12)

                if problem_type == "classification":

                    graph_title = "Machine Learning Model Accuracy Comparison"

                else:

                    graph_title = "Machine Learning Model R2 Score Comparison"

                plt.title(
                    graph_title,
                    fontsize=16,
                    weight="bold",
                )

                plt.grid(axis="x", linestyle="--", alpha=0.4)

                for bar, value in zip(bars, graph_values):

                    plt.text(
                        value + 0.5,
                        bar.get_y() + bar.get_height() / 2,
                        f"{value:.2f}%",
                        va="center",
                        fontsize=10,
                        weight="bold",
                    )

                plt.tight_layout()

                from django.utils import timezone

                graph_file = (
                    f"model_accuracy_{timezone.now().strftime('%Y%m%d_%H%M%S')}.png"
                )

                plt.savefig(
                    os.path.join(
                        graph_dir,
                        graph_file,
                    ),
                    dpi=200,
                    bbox_inches="tight",
                )

                plt.close("all")

                graph_url = settings.MEDIA_URL + "ml_graphs/" + graph_file

                # ============================
                # Save Everything in Session
                # ============================

                request.session["ml_model_results"] = list(graph_results)

                # request.session["ml_best_model"] = {"model": best_model["model"]}
                request.session["ml_best_model"] = best_model
                best_model = request.session["ml_best_model"]

                request.session["ml_graph_url"] = graph_url
                model_results = request.session["ml_model_results"]
                best_model = request.session["ml_best_model"]
                graph_url = request.session["ml_graph_url"]

                dataset_loaded = True

        # =====================================================
        # STEP 3 : MANUAL PREDICTION
        # =====================================================

        elif request.POST.get("manual_predict"):

            prediction = None

            # ------------------------------------
            # Validate dataset/model availability
            # ------------------------------------

            temp_file = request.session.get("ml_dataset")
            features = request.session.get("ml_features")
            selected_model_name = ""

            if best_model:
                selected_model_name = best_model.get("model", "")

            problem_type = request.session.get("ml_problem_type", "classification")
            model_results = request.session.get("ml_model_results", [])

            best_model = request.session.get("ml_best_model", {})

            selected_model_name = ""

            if best_model:
                selected_model_name = best_model.get("model", "")

            prediction_fields = request.session.get("ml_features", [])

            if not temp_file or not os.path.exists(temp_file):
                prediction = ["Please upload a dataset first."]

            elif not best_model:
                prediction = ["Please train a model first."]

            elif not features:
                prediction = ["Model information is missing. Please train again."]

            else:

                try:

                    model_path = os.path.join(
                        settings.MEDIA_ROOT,
                        "ml_models",
                        "model_package.pkl",
                    )

                    if not os.path.exists(model_path):
                        prediction = [
                            "No trained model found. Please train a model first."
                        ]
                        raise Exception("No model")

                    with open(model_path, "rb") as f:
                        package = pickle.load(f)

                    selected_model = request.POST.get("prediction_model")

                    if not selected_model:

                        selected_model = package["best_model"]

                    model = package["models"][selected_model]
                    scaler = package.get("scaler")
                    features = package["features"]
                    encoders = package["encoders"]
                    target_encoder = package.get("target_encoder")

                    input_data = {}

                    # -----------------------------
                    # Validate all inputs
                    # -----------------------------

                    for feature in features:

                        value = request.POST.get(feature, "").strip()

                        if value == "":
                            prediction = [f"Please enter value for '{feature}'."]
                            raise Exception("Missing input")

                        input_data[feature] = value

                    X_new = pd.DataFrame([input_data])

                    # Encode categorical columns

                    for col, encoder in encoders.items():

                        if col in X_new.columns:

                            try:
                                X_new[col] = encoder.transform(X_new[col].astype(str))

                            except ValueError:

                                allowed = ", ".join(map(str, encoder.classes_[:10]))

                                prediction = [
                                    f"'{X_new[col].iloc[0]}' is not present in the training dataset for '{col}'. "
                                    f"Please use one of these values: {allowed}"
                                ]

                                raise Exception("Encoding Error")

                    feature_types = package.get("feature_types", {})

                    for col in X_new.columns:

                        if "datetime" in feature_types.get(col, ""):

                            X_new[col] = pd.to_datetime(
                                X_new[col],
                                errors="coerce",
                            ).astype("int64")

                    for col in X_new.columns:

                        X_new[col] = pd.to_numeric(
                            X_new[col],
                            errors="coerce",
                        )

                    X_new = X_new.fillna(0)
                    X_new = X_new.replace([np.inf, -np.inf], 0)

                    # ----------------------------------------
                    # Scale input exactly like training data
                    # ----------------------------------------

                    if scaler is not None:

                        X_new = pd.DataFrame(
                            scaler.transform(X_new),
                            columns=X_new.columns,
                        )

                    result = model.predict(X_new)

                    if target_encoder is not None:
                        result = target_encoder.inverse_transform(result)

                    prediction = result.tolist()

                except Exception as e:

                    if prediction is None:
                        prediction = ["Prediction Error : " + str(e)]

    # =====================================================
    # FINAL CONTEXT
    # =====================================================

    context = {
        "form": form,
        "dataset_loaded": dataset_loaded,
        "columns": columns,
        "summary": summary,
        "model_results": model_results,
        "graph_url": graph_url,
        "best_model": best_model,
        "prediction": prediction,
        "preview_columns": preview_columns,
        "preview_data": preview_data,
        "prediction_fields": prediction_fields,
        "problem_type": problem_type,
        "selected_model_name": selected_model_name,
        "model_explanation": model_explanation,
    }

    return render(
        request,
        "core/model_training.html",
        context,
    )


# =============================================================================================================================


@login_required
def data_visualization(request):

    form = DatasetUploadForm()
    dataset_loaded = False
    summary = {}
    preview_columns = []
    preview_data = []
    charts = []

    if request.method == "POST":

        if "dataset" in request.FILES:
            form = DatasetUploadForm(request.POST, request.FILES)
        else:
            form = DatasetUploadForm()

        if request.FILES or request.session.get("temp_dataset"):

            try:

                ####################################################
                # LOAD DATASET
                ####################################################

                if "dataset" in request.FILES:

                    uploaded_file = request.FILES["dataset"]

                    df = read_dataset(uploaded_file)

                    temp_folder = os.path.join(
                        settings.MEDIA_ROOT,
                        "temp",
                    )

                    os.makedirs(temp_folder, exist_ok=True)

                    filename = f"{uuid.uuid4()}.csv"

                    path = os.path.join(
                        temp_folder,
                        filename,
                    )

                    df.to_csv(path, index=False)

                    request.session["temp_dataset"] = filename

                else:

                    filename = request.session.get("temp_dataset")

                    path = os.path.join(
                        settings.MEDIA_ROOT,
                        "temp",
                        filename,
                    )

                    df = pd.read_csv(path)

                ####################################################
                # BASIC INFO
                ####################################################

                dataset_loaded = True

                summary = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "missing": int(df.isna().sum().sum()),
                    "duplicates": int(df.duplicated().sum()),
                }

                PREVIEW_ROWS = 200

                preview_columns = df.columns.tolist()

                preview_data = (
                    df.head(PREVIEW_ROWS).fillna("").astype(str).values.tolist()
                )

                ####################################################
                # FIND CATEGORICAL COLUMNS (WORKS FOR ANY DATASET)
                ####################################################

                categorical_columns = []

                for col in df.columns:

                    try:

                        unique = df[col].nunique(dropna=True)

                        # Ignore columns with only one value
                        if unique < 2:
                            continue

                        # Text / Category / Boolean columns
                        if (
                            df[col].dtype == "object"
                            or str(df[col].dtype) == "category"
                            or str(df[col].dtype) == "bool"
                        ):
                            categorical_columns.append(col)
                            continue

                        # Numeric columns with few unique values
                        if pd.api.types.is_numeric_dtype(df[col]):

                            # Treat as categorical only if number of unique values
                            # is small compared to dataset size.
                            if unique <= min(20, max(5, int(len(df) * 0.05))):
                                categorical_columns.append(col)

                    except Exception:
                        pass

                print("Categorical Columns:", categorical_columns)

                ####################################################
                # GENERATE PIE CHARTS
                ####################################################

                for col in categorical_columns:

                    counts = df[col].fillna("Missing").astype(str).value_counts()

                    fig = px.pie(
                        values=counts.values,
                        names=counts.index,
                        # title=f"{col} Distribution",
                        hole=0.35,
                    )

                    fig.update_traces(
                        textposition="inside",
                        textinfo="percent+label",
                    )

                    fig.update_layout(
                        template="plotly_white",
                        height=500,
                        paper_bgcolor="white",
                        plot_bgcolor="white",
                        font=dict(size=14),
                        title=dict(x=0.5, xanchor="center", font=dict(size=22)),
                        legend=dict(orientation="v", x=1.02, y=0.5),
                        margin=dict(
                            l=30,
                            r=30,
                            t=70,
                            b=30,
                        ),
                    )

                    largest = counts.idxmax()

                    largest_percent = round(
                        counts.max() / counts.sum() * 100,
                        1,
                    )

                    top_count = counts.max()
                    total = counts.sum()

                    description = [
                        {
                            "title": "Categorical Feature Analysis",
                            "text": (
                                f"The pie chart provides a proportional view of category distribution "
                                f"within the '{col}' feature. It highlights how records are divided "
                                f"among different groups present in the dataset."
                            ),
                        },
                        {
                            "title": "Dataset Coverage",
                            "text": (
                                f"A total of {total:,} records were analysed. "
                                f"Each segment represents the contribution of an individual category "
                                f"towards the complete dataset."
                            ),
                        },
                        {
                            "title": "Dominant Segment",
                            "text": (
                                f"The category '{largest}' is the most represented group with "
                                f"{top_count:,} records, contributing approximately {largest_percent}% "
                                f"of all observations."
                            ),
                        },
                        {
                            "title": "Data Interpretation",
                            "text": (
                                "This visualization helps identify dominant groups, category imbalance "
                                "and unusual distributions. Highly concentrated categories may require "
                                "special handling during statistical analysis or machine learning "
                                "model preparation."
                            ),
                        },
                    ]

                    charts.append(
                        {
                            "title": f"{col} Distribution",
                            "type": "pie",
                            "description": description,
                            "graph": fig.to_json(),
                        }
                    )

                ####################################################
                # GENERATE TREEMAP VISUALIZATION
                ####################################################

                try:

                    for col in categorical_columns:

                        try:

                            counts = (
                                df[col]
                                .fillna("Missing")
                                .astype(str)
                                .value_counts()
                                .reset_index()
                            )

                            counts.columns = ["Category", "Count"]

                            if len(counts) > 30:

                                counts = counts.head(30)

                            fig = px.treemap(
                                counts,
                                path=["Category"],
                                values="Count",
                            )

                            fig.update_layout(
                                template="plotly_white",
                                height=650,
                                paper_bgcolor="white",
                                plot_bgcolor="#FAFAFA",
                                margin=dict(l=40, r=40, t=50, b=40),
                            )

                            top_category = counts.iloc[0]["Category"]

                            top_value = counts.iloc[0]["Count"]

                            description = [
                                {
                                    "title": "Hierarchical Category Analysis",
                                    "text": (
                                        f"The treemap visualizes the distribution of categories within "
                                        f"'{col}' using proportional areas. Larger blocks represent categories "
                                        "with higher record frequency."
                                    ),
                                },
                                {
                                    "title": "Largest Category Contribution",
                                    "text": (
                                        f"The category '{top_category}' represents the largest segment "
                                        f"with {top_value:,} records, indicating a significant share of "
                                        "the overall dataset."
                                    ),
                                },
                                {
                                    "title": "Visualization Purpose",
                                    "text": (
                                        "Treemaps are useful for comparing multiple categories simultaneously "
                                        "because they display relative importance through visual area sizing."
                                    ),
                                },
                                {
                                    "title": "Analytical Insight",
                                    "text": (
                                        "This analysis helps identify dominant groups, smaller segments and "
                                        "potential category imbalance that may influence reporting accuracy "
                                        "or machine learning model performance."
                                    ),
                                },
                            ]

                            charts.append(
                                {
                                    "title": f"{col} Category Treemap",
                                    "type": "treemap",
                                    "description": description,
                                    "graph": fig.to_json(),
                                }
                            )

                        except Exception as e:

                            print("Treemap Error", e)

                except Exception as e:

                    print("Treemap Generation Error", e)

                ####################################################
                # GENERATE SUNBURST CHART
                ####################################################

                try:

                    if len(categorical_columns) >= 2:

                        parent_col = categorical_columns[0]

                        child_col = categorical_columns[1]

                        sun_df = (
                            df[[parent_col, child_col]].fillna("Missing").astype(str)
                        )

                        sun_df["Count"] = 1

                        sun_df = (
                            sun_df.groupby([parent_col, child_col]).sum().reset_index()
                        )

                        if len(sun_df) > 200:

                            sun_df = sun_df.head(200)

                        fig = px.sunburst(
                            sun_df, path=[parent_col, child_col], values="Count"
                        )

                        fig.update_layout(
                            template="plotly_white",
                            height=700,
                            paper_bgcolor="white",
                            plot_bgcolor="#FAFAFA",
                            margin=dict(l=50, r=50, t=60, b=50),
                        )

                        description = [
                            {
                                "title": "Multi-Level Category Analysis",
                                "text": (
                                    f"The sunburst chart analyses the relationship between "
                                    f"'{parent_col}' and '{child_col}', showing how detailed categories "
                                    "are distributed inside higher-level groups."
                                ),
                            },
                            {
                                "title": "Hierarchy Understanding",
                                "text": (
                                    "The inner circle represents broader categories while outer layers "
                                    "display detailed sub-categories, allowing users to explore "
                                    "category composition."
                                ),
                            },
                            {
                                "title": "Pattern Discovery",
                                "text": (
                                    "This visualization helps discover hidden relationships, "
                                    "category concentration and contribution patterns across multiple "
                                    "levels of classification."
                                ),
                            },
                            {
                                "title": "Machine Learning Insight",
                                "text": (
                                    "Understanding category hierarchy can support feature engineering, "
                                    "encoding strategy selection and identification of important "
                                    "categorical relationships."
                                ),
                            },
                        ]

                        charts.append(
                            {
                                "title": "Hierarchical Category Sunburst",
                                "type": "sunburst",
                                "description": description,
                                "graph": fig.to_json(),
                            }
                        )

                except Exception as e:

                    print("Sunburst Error", e)

                ####################################################
                # GENERATE BAR CHART / COUNT PLOT
                ####################################################

                try:

                    for col in categorical_columns:

                        try:

                            # Count values
                            counts = (
                                df[col].fillna("Missing").astype(str).value_counts()
                            )

                            # Avoid too many categories
                            if len(counts) > 15:

                                counts = counts.head(15)

                            bar_df = pd.DataFrame(
                                {
                                    "Category": counts.index,
                                    "Count": counts.values,
                                }
                            )

                            fig = px.bar(
                                bar_df,
                                x="Count",
                                y="Category",
                                orientation="h",
                            )

                            fig.update_traces(
                                texttemplate="%{x:,}",
                                textposition="outside",
                                hovertemplate="<b>%{y}</b><br>"
                                "Records: %{x:,}"
                                "<extra></extra>",
                            )

                            fig.update_layout(
                                template="plotly_white",
                                height=600,
                                autosize=True,
                                paper_bgcolor="white",
                                plot_bgcolor="#FAFAFA",
                                xaxis=dict(
                                    title="Number of Records",
                                    showgrid=True,
                                    gridcolor="#E5E7EB",
                                    zeroline=False,
                                ),
                                yaxis=dict(
                                    title=col,
                                    automargin=True,
                                    categoryorder="total ascending",
                                ),
                                font=dict(size=14),
                                margin=dict(l=180, r=80, t=50, b=80),
                            )

                            highest_category = counts.idxmax()

                            highest_count = counts.max()

                            percentage = round(highest_count / counts.sum() * 100, 1)

                            description = [
                                {
                                    "title": "Category Frequency Analysis",
                                    "text": (
                                        f"The bar chart evaluates the frequency distribution of categories "
                                        f"within the '{col}' feature. Each bar represents the number of "
                                        "records belonging to a specific group."
                                    ),
                                },
                                {
                                    "title": "Dataset Representation",
                                    "text": (
                                        f"The visualization compares {counts.sum():,} observations "
                                        "across available categories to identify common and uncommon groups."
                                    ),
                                },
                                {
                                    "title": "Highest Frequency Category",
                                    "text": (
                                        f"The category '{highest_category}' has the highest occurrence "
                                        f"with {highest_count:,} records, representing {percentage}% "
                                        "of the analysed data."
                                    ),
                                },
                                {
                                    "title": "Analytical Interpretation",
                                    "text": (
                                        "Frequency comparison helps detect imbalance, dominant categories "
                                        "and rare groups that may require additional preprocessing."
                                    ),
                                },
                                {
                                    "title": "Machine Learning Consideration",
                                    "text": (
                                        "Highly imbalanced categorical variables can affect model learning "
                                        "and may require encoding adjustments, grouping strategies or "
                                        "sampling techniques."
                                    ),
                                },
                            ]
                            charts.append(
                                {
                                    "title": f"{col} Category Count Analysis",
                                    "type": "bar",
                                    "description": description,
                                    "graph": fig.to_json(),
                                }
                            )

                        except Exception as e:

                            print(f"Bar Chart Error {col}:", e)

                except Exception as e:

                    print("Bar Chart Generation Error:", e)

                ####################################################
                # GENERATE PROFESSIONAL HISTOGRAM CHARTS
                ####################################################

                numeric_columns = df.select_dtypes(
                    include=["int64", "float64", "int32", "float32"]
                ).columns.tolist()

                print("Numeric Columns:", numeric_columns)

                for col in numeric_columns:

                    try:

                        if df[col].nunique(dropna=True) < 5:
                            continue

                        data = df[col].dropna()

                        fig = px.histogram(
                            data,
                            x=col,
                            nbins=30,
                        )

                        mean_value = data.mean()
                        median_value = data.median()

                        # Mean line
                        fig.add_vline(
                            x=mean_value,
                            line_dash="dash",
                            annotation_text=f"Mean: {mean_value:.2f}",
                            annotation_position="top",
                        )

                        # Median line
                        fig.add_vline(
                            x=median_value,
                            line_dash="dot",
                            annotation_text=f"Median: {median_value:.2f}",
                            annotation_position="bottom",
                        )

                        fig.update_traces(
                            marker_line_width=1,
                            opacity=0.85,
                            hovertemplate="<b>%{x}</b><br>"
                            + "Records: %{y}<extra></extra>",
                        )

                        fig.update_layout(
                            template="plotly_white",
                            height=600,
                            paper_bgcolor="white",
                            plot_bgcolor="#FAFAFA",
                            bargap=0.08,
                            # title=dict(
                            #     text=f"{col} Distribution Analysis",
                            #     x=0.5,
                            #     xanchor="center",
                            #     font=dict(size=24),
                            # ),
                            xaxis=dict(
                                title=col,
                                showgrid=False,
                            ),
                            yaxis=dict(
                                title="Number of Records",
                                gridcolor="#E5E7EB",
                            ),
                            font=dict(
                                size=14,
                            ),
                            margin=dict(l=180, r=80, t=80, b=80),
                        )

                        description = [
                            {
                                "title": "Column Analysed",
                                "text": f"The histogram represents the distribution pattern of numerical values in the {col} column.",
                            },
                            {
                                "title": "Total Records",
                                "text": f"The analysis contains {data.count():,} valid numerical records.",
                            },
                            {
                                "title": "Average Value",
                                "text": f"The average value of this column is {mean_value:.2f}.",
                            },
                            {
                                "title": "Median Value",
                                "text": f"The median value of this distribution is {median_value:.2f}.",
                            },
                            {
                                "title": "Distribution Analysis",
                                "text": "The histogram shows frequency distribution, spread, concentration, skewness and possible outliers. Mean and median markers indicate central tendency.",
                            },
                        ]

                        charts.append(
                            {
                                "title": f"{col} Histogram",
                                "type": "histogram",
                                "description": description,
                                "graph": fig.to_json(),
                            }
                        )

                    except Exception as e:

                        print(f"Histogram Error {col}:", e)

                ####################################################
                # GENERATE KDE DENSITY PLOTS
                ####################################################

                try:

                    for col in numeric_columns:

                        try:

                            if df[col].nunique() < 10:
                                continue

                            data = df[col].dropna()

                            fig = px.histogram(
                                df,
                                x=col,
                                marginal="violin",
                                histnorm="density",
                                opacity=0.6,
                            )

                            mean_value = data.mean()
                            median_value = data.median()

                            # Mean Line
                            fig.add_vline(
                                x=mean_value,
                                line_dash="dash",
                                annotation_text=f"Mean<br>{mean_value:.2f}",
                                annotation_position="top right",
                                annotation_font_size=12,
                            )

                            # Median Line
                            fig.add_vline(
                                x=median_value,
                                line_dash="dot",
                                annotation_text=f"Median<br>{median_value:.2f}",
                                annotation_position="top left",
                                annotation_font_size=12,
                            )

                            fig.update_layout(
                                template="plotly_white",
                                height=600,
                                paper_bgcolor="white",
                                plot_bgcolor="#FAFAFA",
                                xaxis_title=col,
                                yaxis_title="Density",
                            )

                            description = [
                                {
                                    "title": "Column Analysed",
                                    "text": f"Density distribution analysis performed for {col}.",
                                },
                                {
                                    "title": "Mean",
                                    "text": f"Average value is {mean_value:.2f}.",
                                },
                                {
                                    "title": "Median",
                                    "text": f"Median value is {median_value:.2f}.",
                                },
                                {
                                    "title": "Distribution Insight",
                                    "text": "Density plot highlights skewness, concentration areas and distribution shape.",
                                },
                            ]

                            charts.append(
                                {
                                    "title": f"{col} Density Analysis",
                                    "type": "density",
                                    "description": description,
                                    "graph": fig.to_json(),
                                }
                            )

                        except Exception as e:
                            print("Density Error", e)

                except Exception as e:
                    print("Density Generation Error", e)

                ####################################################
                # GENERATE LINE CHARTS (TREND ANALYSIS)
                ####################################################

                try:

                    date_columns = []

                    for col in df.columns:

                        try:

                            converted = pd.to_datetime(df[col], errors="coerce")

                            if converted.notna().sum() > len(df) * 0.5:
                                date_columns.append(col)

                        except:
                            pass

                    for date_col in date_columns:

                        try:

                            temp_df = df.copy()

                            temp_df[date_col] = pd.to_datetime(
                                temp_df[date_col], errors="coerce"
                            )

                            temp_df = temp_df.dropna(subset=[date_col])

                            temp_df["Date"] = temp_df[date_col].dt.date

                            numeric_cols = df.select_dtypes(
                                include=["int64", "float64", "int32", "float32"]
                            ).columns.tolist()

                            for value_col in numeric_cols:

                                daily_df = (
                                    temp_df.groupby("Date")[value_col]
                                    .sum()
                                    .reset_index()
                                )

                                if len(daily_df) < 5:
                                    continue

                                fig = px.line(
                                    daily_df,
                                    x="Date",
                                    y=value_col,
                                    markers=True,
                                )

                                fig.update_traces(
                                    line_width=3,
                                    marker_size=8,
                                    hovertemplate="<b>Date:</b> %{x}<br>"
                                    + f"<b>{value_col}:</b> %{{y:,.2f}}"
                                    + "<extra></extra>",
                                )

                                fig.update_layout(
                                    template="plotly_white",
                                    height=600,
                                    paper_bgcolor="white",
                                    plot_bgcolor="#FAFAFA",
                                    xaxis=dict(title="Date", showgrid=False),
                                    yaxis=dict(title=value_col, gridcolor="#E5E7EB"),
                                    margin=dict(l=80, r=60, t=60, b=80),
                                )

                                description = [
                                    {
                                        "title": "Time-Series Analysis",
                                        "text": (
                                            f"The line chart examines how '{value_col}' changes over time "
                                            f"using the '{date_col}' dimension. It highlights movement, "
                                            "growth patterns and fluctuations."
                                        ),
                                    },
                                    {
                                        "title": "Observation Period",
                                        "text": (
                                            f"The analysis covers {len(daily_df):,} time intervals, "
                                            "providing visibility into historical behaviour."
                                        ),
                                    },
                                    {
                                        "title": "Trend Interpretation",
                                        "text": (
                                            "The visualization helps identify upward trends, declining periods, "
                                            "seasonality and unusual spikes that may require investigation."
                                        ),
                                    },
                                    {
                                        "title": "Data Science Insight",
                                        "text": (
                                            "Time-based patterns can be valuable for forecasting models, "
                                            "feature engineering and predictive analytics applications."
                                        ),
                                    },
                                ]

                                charts.append(
                                    {
                                        "title": f"{value_col} Trend Over Time",
                                        "type": "line",
                                        "description": description,
                                        "graph": fig.to_json(),
                                    }
                                )

                        except Exception as e:

                            print("Line Chart Error:", e)

                except Exception as e:

                    print("Line Generation Error:", e)

                ####################################################
                # GENERATE AREA CHARTS (TREND VOLUME ANALYSIS)
                ####################################################

                try:

                    date_columns = []

                    for col in df.columns:

                        try:

                            # Ignore numeric columns completely
                            if pd.api.types.is_numeric_dtype(df[col]):
                                continue

                            converted = pd.to_datetime(
                                df[col], errors="coerce", infer_datetime_format=True
                            )

                            if converted.notna().sum() >= len(df) * 0.70:
                                date_columns.append(col)

                        except:
                            pass

                    for date_col in date_columns:

                        try:

                            temp_df = df.copy()

                            temp_df[date_col] = pd.to_datetime(
                                temp_df[date_col], errors="coerce"
                            )

                            temp_df = temp_df.dropna(subset=[date_col])

                            temp_df["Date"] = temp_df[date_col].dt.date

                            # -------------------------------------------------
                            # Only genuine numeric columns
                            # -------------------------------------------------

                            numeric_cols = []

                            for col in temp_df.columns:

                                # Skip the X-axis date column
                                if col == date_col or col == "Date":
                                    continue

                                # Skip datetime columns completely
                                if pd.api.types.is_datetime64_any_dtype(temp_df[col]):
                                    continue

                                # Keep only numeric columns
                                if pd.api.types.is_numeric_dtype(temp_df[col]):
                                    numeric_cols.append(col)

                            for value_col in numeric_cols:

                                # Skip columns that cannot be summed
                                try:
                                    if not pd.api.types.is_numeric_dtype(
                                        temp_df[value_col]
                                    ):
                                        continue
                                except:
                                    continue

                                area_df = temp_df.groupby("Date", as_index=False)[
                                    value_col
                                ].agg("sum")

                                # Avoid small meaningless charts
                                if len(area_df) < 5:
                                    continue

                                fig = px.area(
                                    area_df,
                                    x="Date",
                                    y=value_col,
                                )

                                total_value = area_df[value_col].sum()

                                max_value = area_df[value_col].max()

                                max_date = area_df.loc[
                                    area_df[value_col].idxmax(), "Date"
                                ]

                                avg_value = area_df[value_col].mean()

                                fig.update_traces(
                                    line_width=3,
                                    hovertemplate="<b>Date:</b> %{x}<br>"
                                    + f"<b>{value_col}:</b> %{{y:,.2f}}"
                                    + "<extra></extra>",
                                )

                                fig.update_layout(
                                    template="plotly_white",
                                    height=600,
                                    paper_bgcolor="white",
                                    plot_bgcolor="#FAFAFA",
                                    xaxis=dict(
                                        title="Date",
                                        showgrid=False,
                                    ),
                                    yaxis=dict(
                                        title=value_col,
                                        gridcolor="#E5E7EB",
                                    ),
                                    margin=dict(
                                        l=80,
                                        r=60,
                                        t=60,
                                        b=80,
                                    ),
                                )

                                description = [
                                    {
                                        "title": "Area Trend Analysis",
                                        "text": (
                                            f"The area chart analyses cumulative movement of "
                                            f"'{value_col}' over time using the '{date_col}' field. "
                                            "The filled region highlights volume changes and growth patterns."
                                        ),
                                    },
                                    {
                                        "title": "Time Coverage",
                                        "text": (
                                            f"The visualization contains {len(area_df):,} time periods "
                                            "showing historical changes in the selected metric."
                                        ),
                                    },
                                    {
                                        "title": "Total Contribution",
                                        "text": (
                                            f"The total accumulated value across the analysed period "
                                            f"is {total_value:,.2f}."
                                        ),
                                    },
                                    {
                                        "title": "Peak Performance",
                                        "text": (
                                            f"The highest recorded value was {max_value:,.2f} "
                                            f"on {max_date}."
                                        ),
                                    },
                                    {
                                        "title": "Data Science Insight",
                                        "text": (
                                            "Area charts are useful for understanding cumulative "
                                            "growth, seasonal behaviour, volume trends and forecasting "
                                            "patterns before applying machine learning models."
                                        ),
                                    },
                                ]

                                charts.append(
                                    {
                                        "title": f"{value_col} Area Trend Analysis",
                                        "type": "area",
                                        "description": description,
                                        "graph": fig.to_json(),
                                    }
                                )

                        except Exception as e:

                            print("Area Chart Error:", e)

                except Exception as e:

                    print("Area Generation Error:", e)

                ####################################################
                # GENERATE BOX PLOTS (OUTLIER ANALYSIS)
                ####################################################

                try:

                    numeric_columns = df.select_dtypes(
                        include=["int64", "float64", "int32", "float32"]
                    ).columns.tolist()

                    for col in numeric_columns:

                        try:

                            # Skip low variation columns
                            if df[col].nunique(dropna=True) < 5:
                                continue

                            data = df[[col]].dropna()

                            fig = px.box(
                                data,
                                y=col,
                                points="outliers",
                            )

                            q1 = data[col].quantile(0.25)
                            q3 = data[col].quantile(0.75)

                            median = data[col].median()

                            iqr = q3 - q1

                            lower_limit = q1 - 1.5 * iqr
                            upper_limit = q3 + 1.5 * iqr

                            outliers = data[
                                (data[col] < lower_limit) | (data[col] > upper_limit)
                            ]

                            fig.update_traces(
                                marker=dict(size=8),
                                hovertemplate="<b>Value:</b> %{y}<extra></extra>",
                            )

                            fig.update_layout(
                                template="plotly_white",
                                height=500,
                                paper_bgcolor="white",
                                plot_bgcolor="#FAFAFA",
                                yaxis=dict(title=col, gridcolor="#E5E7EB"),
                                xaxis=dict(showticklabels=False),
                                margin=dict(
                                    l=60,
                                    r=40,
                                    t=60,
                                    b=60,
                                ),
                            )

                            description = [
                                {
                                    "title": "Column Analysed",
                                    "text": f"The box plot analyses the spread and variability of the {col} numerical feature.",
                                },
                                {
                                    "title": "Median Value",
                                    "text": f"The median value of this column is {median:.2f}.",
                                },
                                {
                                    "title": "Interquartile Range",
                                    "text": f"The middle 50% of observations are distributed between {q1:.2f} and {q3:.2f}.",
                                },
                                {
                                    "title": "Outlier Detection",
                                    "text": f"The analysis identified {len(outliers):,} possible outlier records outside the normal range.",
                                },
                                {
                                    "title": "ML Insight",
                                    "text": "Outliers can influence machine learning models and may require scaling, transformation or investigation.",
                                },
                            ]

                            charts.append(
                                {
                                    "title": f"{col} Outlier Analysis",
                                    "type": "box",
                                    "description": description,
                                    "graph": fig.to_json(),
                                }
                            )

                        except Exception as e:

                            print(f"Box Plot Error {col}:", e)

                except Exception as e:

                    print("Box Plot Generation Error:", e)

                    ####################################################
                # GENERATE VIOLIN PLOTS (DISTRIBUTION ANALYSIS)
                ####################################################

                try:

                    numeric_columns = df.select_dtypes(
                        include=["int64", "float64", "int32", "float32"]
                    ).columns.tolist()

                    for col in numeric_columns:

                        try:

                            # Skip low variation columns
                            if df[col].nunique(dropna=True) < 5:
                                continue

                            data = df[[col]].dropna()

                            fig = px.violin(
                                data,
                                y=col,
                                box=True,
                                points="outliers",
                            )

                            mean_value = data[col].mean()
                            median_value = data[col].median()

                            q1 = data[col].quantile(0.25)
                            q3 = data[col].quantile(0.75)

                            fig.update_traces(
                                box_visible=True,
                                meanline_visible=True,
                                hovertemplate="<b>Value:</b> %{y}<extra></extra>",
                            )

                            fig.update_layout(
                                template="plotly_white",
                                height=600,
                                paper_bgcolor="white",
                                plot_bgcolor="#FAFAFA",
                                yaxis=dict(
                                    title=col,
                                    gridcolor="#E5E7EB",
                                ),
                                xaxis=dict(showticklabels=False),
                                margin=dict(
                                    l=80,
                                    r=50,
                                    t=60,
                                    b=60,
                                ),
                            )

                            description = [
                                {
                                    "title": "Distribution Analysis",
                                    "text": (
                                        f"The violin plot analyses the probability distribution "
                                        f"and density pattern of the numerical feature '{col}'. "
                                        "The width of the violin represents data concentration."
                                    ),
                                },
                                {
                                    "title": "Central Tendency",
                                    "text": (
                                        f"The average value is {mean_value:.2f} and the median "
                                        f"value is {median_value:.2f}, representing the central "
                                        "location of the dataset."
                                    ),
                                },
                                {
                                    "title": "Spread Analysis",
                                    "text": (
                                        f"The middle 50% of observations are distributed between "
                                        f"{q1:.2f} and {q3:.2f}, showing the interquartile spread."
                                    ),
                                },
                                {
                                    "title": "Shape Interpretation",
                                    "text": (
                                        "The violin shape helps identify skewness, multiple peaks, "
                                        "concentration regions and differences in data density."
                                    ),
                                },
                                {
                                    "title": "Machine Learning Insight",
                                    "text": (
                                        "Understanding feature distribution helps in selecting "
                                        "appropriate transformations, scaling techniques and "
                                        "preprocessing strategies before model training."
                                    ),
                                },
                            ]

                            charts.append(
                                {
                                    "title": f"{col} Violin Distribution",
                                    "type": "violin",
                                    "description": description,
                                    "graph": fig.to_json(),
                                }
                            )

                        except Exception as e:

                            print(f"Violin Plot Error {col}:", e)

                except Exception as e:

                    print("Violin Generation Error:", e)

                ####################################################
                # GENERATE FEATURE RELATIONSHIP SCATTER PLOTS
                ####################################################

                try:

                    numeric_columns = df.select_dtypes(
                        include=["int64", "float64", "int32", "float32"]
                    ).columns.tolist()

                    numeric_columns = [
                        col
                        for col in numeric_columns
                        if df[col].nunique(dropna=True) > 5
                    ]

                    # Limit number of columns
                    numeric_columns = numeric_columns[:5]

                    print("Scatter Plot Columns:", numeric_columns)

                    import itertools

                    # Generate every possible combination
                    feature_pairs = list(itertools.combinations(numeric_columns, 2))

                    for x_col, y_col in feature_pairs:

                        pair_df = df[[x_col, y_col]].dropna()

                        # Add trend line
                        fig = px.scatter(
                            pair_df,
                            x=x_col,
                            y=y_col,
                            trendline="ols",
                            opacity=0.65,
                        )

                        fig.update_traces(
                            marker=dict(
                                size=8,
                            )
                        )

                        fig.update_layout(
                            template="plotly_white",
                            height=500,
                            paper_bgcolor="white",
                            plot_bgcolor="#FAFAFA",
                            # title=dict(
                            #     text=f"{x_col} vs {y_col}",
                            #     x=0.5,
                            #     xanchor="center",
                            #     font=dict(size=22),
                            # ),
                            xaxis=dict(
                                title=x_col,
                                showgrid=False,
                            ),
                            yaxis=dict(
                                title=y_col,
                                gridcolor="#E5E7EB",
                            ),
                            margin=dict(
                                l=60,
                                r=40,
                                t=80,
                                b=60,
                            ),
                        )

                        correlation = round(pair_df[x_col].corr(pair_df[y_col]), 2)

                        description = [
                            {
                                "title": "Features Analysed",
                                "text": f"The scatter plot analyses the relationship between {x_col} and {y_col}.",
                            },
                            {
                                "title": "Correlation",
                                "text": f"The correlation coefficient between these features is {correlation}.",
                            },
                            {
                                "title": "Relationship Pattern",
                                "text": "The visualization helps identify trends, clusters, linear relationships and unusual observations.",
                            },
                            {
                                "title": "Machine Learning Insight",
                                "text": "Strong correlations may indicate useful predictive features, while weak relationships may provide limited predictive value.",
                            },
                        ]

                        charts.append(
                            {
                                "title": f"{x_col} vs {y_col}",
                                "type": "scatter",
                                "description": description,
                                "graph": fig.to_json(),
                            }
                        )

                except Exception as e:

                    print("Scatter Plot Error:", e)

                    ####################################################
                # GENERATE CORRELATION HEATMAP
                ####################################################

                try:

                    numeric_columns = df.select_dtypes(
                        include=["int64", "float64", "int32", "float32"]
                    ).columns.tolist()

                    # Remove columns with no variation
                    numeric_columns = [
                        col
                        for col in numeric_columns
                        if df[col].nunique(dropna=True) > 1
                    ]

                    if len(numeric_columns) >= 2:

                        corr_df = df[numeric_columns].corr(method="pearson")

                        fig = px.imshow(
                            corr_df,
                            text_auto=".2f",
                            aspect="auto",
                            color_continuous_scale="RdBu",
                            zmin=-1,
                            zmax=1,
                        )

                        fig.update_layout(
                            template="plotly_white",
                            height=700,
                            paper_bgcolor="white",
                            plot_bgcolor="#FAFAFA",
                            xaxis=dict(
                                title="Features",
                                tickangle=-45,
                            ),
                            yaxis=dict(
                                title="Features",
                            ),
                            margin=dict(
                                l=80,
                                r=80,
                                t=60,
                                b=120,
                            ),
                        )

                        # Find strongest correlations

                        corr_matrix = np.triu(np.ones(corr_df.shape), k=1).astype(bool)

                        corr_pairs = (
                            corr_df.where(corr_matrix)
                            .stack()
                            .sort_values(ascending=False)
                        )

                        if len(corr_pairs) > 0:

                            strongest_feature_pair = corr_pairs.index[0]

                            strongest_value = corr_pairs.iloc[0]

                        else:

                            strongest_feature_pair = ("None", "None")

                            strongest_value = 0

                        description = [
                            {
                                "title": "Features Analysed",
                                "text": f"The correlation heatmap analyses relationships between {len(numeric_columns)} numerical features.",
                            },
                            {
                                "title": "Correlation Range",
                                "text": "Correlation values range from -1 to +1. Positive values indicate direct relationships, while negative values indicate inverse relationships.",
                            },
                            {
                                "title": "Strongest Relationship",
                                "text": f"The strongest correlation detected is between {strongest_feature_pair[0]} and {strongest_feature_pair[1]} with a value of {strongest_value:.2f}.",
                            },
                            {
                                "title": "Machine Learning Insight",
                                "text": "Highly correlated features may contain similar information and can be considered for feature selection or dimensionality reduction.",
                            },
                        ]

                        charts.append(
                            {
                                "title": "Feature Correlation Heatmap",
                                "type": "heatmap",
                                "description": description,
                                "graph": fig.to_json(),
                            }
                        )

                except Exception as e:

                    print("Correlation Heatmap Error:", e)

                ####################################################
                # GENERATE PAIR PLOT
                ####################################################

                try:

                    pair_columns = numeric_columns[:5]

                    if len(pair_columns) >= 3:

                        fig = px.scatter_matrix(
                            df, dimensions=pair_columns, opacity=0.5
                        )

                        fig.update_layout(
                            template="plotly_white",
                            height=900,
                            paper_bgcolor="white",
                        )

                        description = [
                            {
                                "title": "Features Analysed",
                                "text": f"Pair plot analyses relationships between {len(pair_columns)} numerical variables.",
                            },
                            {
                                "title": "Purpose",
                                "text": "Helps identify correlation patterns, clusters and abnormal observations.",
                            },
                            {
                                "title": "ML Insight",
                                "text": "Useful for selecting important features before model training.",
                            },
                        ]

                        charts.append(
                            {
                                "title": "Numerical Feature Pair Relationship",
                                "type": "pairplot",
                                "description": description,
                                "graph": fig.to_json(),
                            }
                        )

                except Exception as e:

                    print("Pair Plot Error:", e)

                ####################################################
                # MISSING VALUE ANALYSIS
                ####################################################

                try:

                    missing_df = df.isnull().sum().reset_index()

                    missing_df.columns = ["Column", "Missing"]

                    missing_df = missing_df[missing_df["Missing"] > 0]

                    if len(missing_df) > 0:

                        fig = px.bar(
                            missing_df, x="Column", y="Missing", text="Missing"
                        )

                        fig.update_layout(
                            template="plotly_white", height=500, xaxis_tickangle=-45
                        )

                        description = [
                            {
                                "title": "Missing Data Analysis",
                                "text": "Visualization shows columns containing missing values.",
                            },
                            {
                                "title": "Total Missing Values",
                                "text": f"{missing_df['Missing'].sum():,} missing records detected.",
                            },
                        ]

                        charts.append(
                            {
                                "title": "Missing Value Analysis",
                                "type": "missing",
                                "description": description,
                                "graph": fig.to_json(),
                            }
                        )

                except Exception as e:
                    print("Missing Analysis Error", e)

            except Exception as e:

                form.add_error(
                    "dataset",
                    str(e),
                )
    print("Charts Generated:", len(charts))

    for c in charts:
        print(c["title"])

    context = {
        "form": form,
        "dataset_loaded": dataset_loaded,
        "summary": summary,
        "preview_columns": preview_columns,
        "preview_data": preview_data,
        "preview_rows": PREVIEW_ROWS if dataset_loaded else 0,
        "charts": charts,
    }

    return render(
        request,
        "core/data_visualization.html",
        context,
    )


# ===============================================================================================================================
