from django.urls import include, path, re_path
from rest_framework.urlpatterns import format_suffix_patterns
from maps import views


urlpatterns = [
    # re_path(r'^(?P<map_name>\w+)/$', views.route_map, name='route-map'),
    path("", views.route_map, name='route-map'),
]
urlpatterns = format_suffix_patterns(urlpatterns)
