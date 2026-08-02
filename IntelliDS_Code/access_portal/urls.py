"""
URL configuration for access_portal project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from core.views import (
    handler_400,
    handler401,
    handler403,
    handler404,
    handler500,
    handler502,
    handler503,
)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('core.urls')),  # All core app URLs
]

# === Error Handlers ===
handler400 = handler_400
handler401 = handler401
handler403 = handler403
handler404 = handler404
handler500 = handler500
handler502 = handler502
handler503 = handler503

