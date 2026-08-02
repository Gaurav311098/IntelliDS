from django.urls import path
from core.views import (
    login_view,
    logout_view,
    logout_and_redirect,
    admin_dashboard,
    user_dashboard,
    get_user_menus,
    dynamic_menu_page,
    upload_dataset,
    data_preparation,
    model_training,
    data_visualization
)

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # Authentication
    path('', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('logout-redirect/', logout_and_redirect, name='logout_and_redirect'),

    # Dashboards
    path('admin-dashboard/', admin_dashboard, name='admin_dashboard'),
    path('user-dashboard/', user_dashboard, name='user_dashboard'),

    path("intellids/upload-dataset/",upload_dataset,name="upload_dataset",),
    path("intellids/data-preparation/",data_preparation,name="data_preparation",),
    path("intellids/data-visualization/",data_visualization,name="data_visualization",),
    path("intellids/model-training/",model_training,name="model_training",),

    # AJAX / API
    path('get_user_menus/<int:user_id>/', get_user_menus, name='get_user_menus'),

    # Dynamic menu pages
    path('<str:url_name>/', dynamic_menu_page, name='dynamic_menu_page'),
]

if settings.DEBUG:

    urlpatterns += static(
        settings.MEDIA_URL,
        document_root=settings.MEDIA_ROOT
    )