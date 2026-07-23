from pathlib import Path

from rest_framework import serializers

from matching.models import MatchResult, SubmittedResume

MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".docx", ".txt"}


class MatchOptionsSerializer(serializers.Serializer):
    top_n = serializers.IntegerField(required=False, default=10, min_value=1, max_value=50)
    explain_top_k = serializers.IntegerField(required=False, default=3, min_value=1, max_value=10)


class MatchRequestSerializer(MatchOptionsSerializer):
    resume_text = serializers.CharField(min_length=10, trim_whitespace=True)


class FileUploadSerializer(MatchOptionsSerializer):
    resume_file = serializers.FileField()

    def validate_resume_file(self, value):
        if value.size > MAX_UPLOAD_SIZE:
            raise serializers.ValidationError(
                f"File too large ({value.size / 1024 / 1024:.1f}MB) - max {MAX_UPLOAD_SIZE // 1024 // 1024}MB."
            )
        if Path(value.name).suffix.lower() not in ALLOWED_UPLOAD_EXTENSIONS:
            raise serializers.ValidationError(
                "Unsupported file type. Please upload a .pdf, .docx, or .txt file, "
                "or paste your resume text instead."
            )
        return value


class MatchResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = MatchResult
        fields = ["rank", "job_doc_id", "category", "title", "score", "explanations"]


class SubmittedResumeHistorySerializer(serializers.ModelSerializer):
    matches = MatchResultSerializer(many=True, read_only=True)

    class Meta:
        model = SubmittedResume
        fields = [
            "id",
            "original_filename",
            "submitted_at",
            "ats_score",
            "skill_gap",
            "matches",
        ]
