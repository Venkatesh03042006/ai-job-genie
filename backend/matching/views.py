from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from matching.extraction import TextExtractionError, extract_text
from matching.models import MatchResult, SubmittedResume
from matching.serializers import (
    FileUploadSerializer,
    MatchRequestSerializer,
    SubmittedResumeHistorySerializer,
)
from matching.services import get_matches


def _submit_and_match(*, resume_text, user, top_n, explain_top_k, original_filename=None):
    submitted_resume = SubmittedResume.objects.create(
        user=user,
        resume_text=resume_text,
        original_filename=original_filename,
    )

    result = get_matches(resume_text, top_n=top_n, explain_top_k=explain_top_k)
    matches = result["matches"]

    MatchResult.objects.bulk_create([
        MatchResult(
            submitted_resume=submitted_resume,
            rank=m["rank"],
            job_doc_id=m["job_doc_id"],
            category=m["category"],
            title=m["title"],
            score=m["score"],
            explanations=m["explanations"],
        )
        for m in matches
    ])

    submitted_resume.ats_score = result["ats_score"]
    submitted_resume.skill_gap = result["skill_gap"]
    submitted_resume.save(update_fields=["ats_score", "skill_gap"])

    return {
        "resume_id": submitted_resume.id,
        "matches": matches,
        "ats_score": result["ats_score"],
        "skill_gap": result["skill_gap"],
    }


class MatchView(APIView):
    """POST /api/match/ - rank a pasted resume against the real job posting
    pool and return chunk-level explanations for the top matches (see
    matching/services.py). No auth required yet - AUTH_USER_MODEL FK on
    SubmittedResume is ready for whenever that's wired up.
    """
    permission_classes = [AllowAny]

    def post(self, request):
        request_serializer = MatchRequestSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)
        data = request_serializer.validated_data

        result = _submit_and_match(
            resume_text=data["resume_text"],
            user=request.user if request.user.is_authenticated else None,
            top_n=data["top_n"],
            explain_top_k=data["explain_top_k"],
        )

        return Response(result, status=201)


class MatchFileUploadView(APIView):
    """POST /api/match/upload/ - same matching flow as MatchView, but the
    resume text is extracted from an uploaded PDF/DOCX/TXT file
    (matching/extraction.py) instead of being pasted directly.
    """
    permission_classes = [AllowAny]

    def post(self, request):
        request_serializer = FileUploadSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)
        data = request_serializer.validated_data
        uploaded_file = data["resume_file"]

        try:
            resume_text = extract_text(uploaded_file)
        except TextExtractionError as exc:
            return Response({"resume_file": [str(exc)]}, status=400)

        result = _submit_and_match(
            resume_text=resume_text,
            user=request.user if request.user.is_authenticated else None,
            top_n=data["top_n"],
            explain_top_k=data["explain_top_k"],
            original_filename=uploaded_file.name,
        )

        return Response(result, status=201)


class MatchHistoryView(APIView):
    """GET /api/match/history/ - the logged-in user's past submissions and
    their match results, newest first.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        resumes = (
            SubmittedResume.objects.filter(user=request.user)
            .order_by("-submitted_at")
            .prefetch_related("matches")
        )
        return Response(SubmittedResumeHistorySerializer(resumes, many=True).data)
