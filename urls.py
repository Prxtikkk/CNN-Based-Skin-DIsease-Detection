from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, include


from . import views

urlpatterns = [
    path('', views.index, name="home"),
    path('index/', views.index, name="index"),
    path('classification1', views.classification1, name="classification1"),
    path('register/', views.register, name="register"),
    path('logout_request/', views.logout_request, name="logout_request"),
    path('login1/', views.login1, name="login1"),  
    path('admin/', admin.site.urls),
    path('fruits/',views.fruits,name='fruits'),
    path("about/",views.about,name='about'),

    
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)