from django.urls import path
from . import views
from django.conf.urls import url
from django.conf import settings
from django.conf.urls.static import static

urlpatterns=[
            path('index', views.main_index, name='main_index'),
            ]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)