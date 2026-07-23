from django.urls import path

from matching.views import MatchFileUploadView, MatchHistoryView, MatchView

urlpatterns = [
    path("match/", MatchView.as_view(), name="match"),
    path("match/upload/", MatchFileUploadView.as_view(), name="match-upload"),
    path("match/history/", MatchHistoryView.as_view(), name="match-history"),
]
