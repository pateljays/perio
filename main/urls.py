from django.urls import path
from . import views
from django.conf.urls import url
from django.conf import settings
from django.conf.urls.static import static

urlpatterns=[
            path('', views.main_index, name='main_index'),
            url('api/get_pid/', views.get_pid, name='get_pid')
            ]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)