<xsl:stylesheet
	version="2.0"
	xmlns="http://www.w3.org/1999/xhtml"
	xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

<xsl:include href="M.xsl"/>
<xsl:include href="I.xsl"/>
<xsl:include href="H.xsl"/>

<xsl:template match="/">
    <xsl:result-document href="qresultsW.html">
	<html>
	<head>
		<title>MustAlias: Proven by Worst Case</title>
		<link rel="stylesheet" href="style.css" type="text/css"/>
	</head>
	<body>
	<table class="details">
    <colgroup>
            <col width="50%"/>
            <col width="50%"/>
    </colgroup>
	<xsl:for-each select="qresults/queries/query">
			<tr><td><xsl:apply-templates select="id(@Iid)"/></td>
				<td><xsl:apply-templates select="id(@Hid)"/></td>
			</tr>
	</xsl:for-each>
	</table>
	</body>
	</html>
    </xsl:result-document>
</xsl:template>

</xsl:stylesheet>

